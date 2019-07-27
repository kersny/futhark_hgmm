import "lib/github.com/diku-dk/linalg/linalg"
import "lib/github.com/athas/vector/vspace"
module v2 = mk_vspace_2d f32
module v3 = mk_vspace_3d f32

type ftype = f32

-- ftype
module ftypelinalg = mk_linalg f32

type intrinsics = {fx:ftype, fy:ftype, cx:ftype, cy:ftype}
type iso3d = {R: [3][3]ftype, t: v3.vector}
type m33 = {r11:ftype, r12:ftype, r13:ftype,
            r21:ftype, r22:ftype, r23:ftype,
            r31:ftype, r32:ftype, r33:ftype}
let m33_to_mat (m: m33): [][]ftype =
    [[m.r11, m.r12, m.r13],
     [m.r21, m.r22, m.r23],
     [m.r31, m.r32, m.r33]]
let mat_to_m33 (m: [][]ftype): m33 =
    {r11= m[0,0], r12= m[0,1], r13= m[0,2],
     r21= m[1,0], r22= m[1,1], r23= m[1,2],
     r31= m[2,0], r32= m[2,1], r33= m[2,2]}

let m33add (ma: m33) (mb: m33) : m33 =
    {r11= ma.r11 + mb.r11, r12= ma.r12 + mb.r12, r13= ma.r13 + mb.r13,
     r21= ma.r21 + mb.r21, r22= ma.r22 + mb.r22, r23= ma.r23 + mb.r23,
     r31= ma.r31 + mb.r31, r32= ma.r32 + mb.r32, r33= ma.r33 + mb.r33}

let m33sub (ma: m33) (mb: m33) : m33 =
    {r11= ma.r11 - mb.r11, r12= ma.r12 - mb.r12, r13= ma.r13 - mb.r13,
     r21= ma.r21 - mb.r21, r22= ma.r22 - mb.r22, r23= ma.r23 - mb.r23,
     r31= ma.r31 - mb.r31, r32= ma.r32 - mb.r32, r33= ma.r33 - mb.r33}

let m33div (ma: m33) (b: ftype) : m33 =
    {r11= ma.r11/b, r12= ma.r12/b, r13= ma.r13/b,
     r21= ma.r21/b, r22= ma.r22/b, r23= ma.r23/b,
     r31= ma.r31/b, r32= ma.r32/b, r33= ma.r33/b}

let m33mul (ma: m33) (b: ftype) : m33 =
    {r11= ma.r11*b, r12= ma.r12*b, r13= ma.r13*b,
     r21= ma.r21*b, r22= ma.r22*b, r23= ma.r23*b,
     r31= ma.r31*b, r32= ma.r32*b, r33= ma.r33*b}


let unproject(intr: intrinsics, pt: v2.vector, depth: ftype): v3.vector =
   let x = pt.x/intr.fx - intr.cx/intr.fx in
   let y = pt.y/intr.fy - intr.cy/intr.fy in
   {x=x*depth,y=y*depth,z=depth}

let transform(tf: iso3d, pt: v3.vector): v3.vector =
    let pa = [pt.x, pt.y, pt.z] in
    let rr = ftypelinalg.matvecmul_row(tf.R)(pa) in
    let pr = {x=rr[0], y=rr[1], z=rr[2]} in
    pr v3.+ tf.t


entry depth_to_pointcloud [n][m] (im: [n][m]u16, intr: intrinsics, tf_to_world: iso3d): []v3.vector =
  flatten (map(\row ->
    map(\col ->
        let dvr = im[row,col] in
        -- ftype
        let dvf = f32.u16 dvr in
        let dvs = dvf/(f32.f64 5000.0) in
        let pt_cam = unproject(intr, {x=f32.i32 row,y=f32.i32 col}, dvs) in
        transform(tf_to_world, pt_cam)
        )
        (iota m)
    )
    (iota n))

let intpc [n] (pc: [n]v3.vector, res: ftype): [][]i32 =
    map(\vf ->
        let xr = vf.x/res in
        let yr = vf.y/res in
        let zr = vf.z/res in
        let xi = f32.to_i32 xr in
        let yi = f32.to_i32 yr in
        let zi = f32.to_i32 zr in
        [xi, yi, zi]) pc

-- Store elements for which bitn is not set first
let rs_step_asc [n] ((xs:[n]u32,is:[n]i32),bitn:i32) : ([n]u32,[n]i32) =
  let bits1 = map (u32.get_bit bitn) xs
  --let bits1 = map (\x -> (i32.u32 x >>> bitn) & 1) xs
  let bits0 = map (1-) bits1
  let idxs0 = map2 (*) bits0 (scan (+) 0 bits0)
  let idxs1 = scan (+) 0 bits1
  let offs  = reduce (+) 0 bits0    -- store idxs1 last
  let idxs1 = map2 (*) bits1 (map (+offs) idxs1)
  let idxs  = map (\x->x-1) (map2 (+) idxs0 idxs1)
  in (scatter (copy xs) idxs xs,
      scatter (copy is) idxs is)

-- Radix sort - ascending
let rsort_asc [n] (xs: [n]u32) : ([n]u32,[n]i32) =
  let is = iota n
  in loop (p : ([n]u32,[n]i32)) = (xs,is) for i < 32 do
    rs_step_asc(p,i)

let sgm_scan_add [n] (vals:[n]i32) (flags:[n]bool) : [n]i32 =
    let pairs = scan ( \ (v1,f1) (v2,f2) ->
                       let f = f1 || f2
                       let v = if f2 then v2 else v1 + v2
                       in (v,f) ) (0,false) (zip vals flags)
    let (ret,_) = unzip pairs
    in ret

let sgm_scan_addv [n] (vals:[n]v3.vector) (flags:[n]bool) : [n]v3.vector =
    let f32z = f32.f64 0.0
    let pairs = scan ( \ (va,f1) (vb,f2) ->
                       let f = f1 || f2
                       let v = if f2 then vb else va v3.+ vb
                       in (v,f) ) ({x=f32z,y=f32z,z=f32z},false) (zip vals flags)
    let (ret,_) = unzip pairs
    in ret

let sgm_scan_addm [n] (vals:[n]m33) (flags:[n]bool) : [n]m33 =
    let zmat = mat_to_m33 (replicate 3 (replicate 3 (f32.f64 0.0)))
    let pairs = scan ( \ (va,f1) (vb,f2) ->
                       let f = f1 || f2
                       let v = if f2 then vb else (m33add va vb)
                       in (v,f) ) (zmat,false) (zip vals flags)
    let (ret,_) = unzip pairs
    in ret

let v3_outerproduct (va:v3.vector, vb: v3.vector) : [3][3]ftype =
    [[va.x*vb.x, va.x*vb.y, va.x*vb.z],
    [va.y*vb.x, va.y*vb.y, va.y*vb.z],
    [va.z*vb.x, va.z*vb.y, va.z*vb.z]]

let fill (va:v3.vector) (vb: v3.vector) : v3.vector =
    let f32z = f32.f64 0.0 in
    if vb.x==f32z && vb.y==f32z && vb.z==f32z then va else vb

entry voxel_grid_downsample [n] (pcr: [n]v3.vector, res:ftype): ([]v3.vector,[][][]ftype) =
    let pc = intpc (pcr,res) in
    let hashed = map(\vf ->
        let x = u32.i32 vf[0] in
        let y = u32.i32 vf[1] in
        let z = u32.i32 vf[2] in
        let andr = u32.i32 0x3ff in
        let s1 = u32.i32 20 in
        let s2 = u32.i32 10 in
        let hash = ((x & andr) << s1) | ((y & andr) << s2) | (z & andr) in
        hash) pc
        in
    let (xs, indices) = rsort_asc(hashed)
    let ys = rotate (-1) xs
    let is_change = (map2 (\x y -> x!=y) xs ys)
    let sorted_pts = scatter (copy pcr) indices pcr
    let ones = replicate n 1
    let cts = sgm_scan_add ones is_change
    let sel = map (\(c,q) -> c>=9 && q) (zip (rotate (-1) cts) is_change)
    let psums = sgm_scan_addv sorted_pts is_change
    let pavgs = map2 (\vf ct -> {x=vf.x/f32.i32(ct),y=vf.y/f32.i32(ct),z=vf.z/f32.i32(ct)}) psums cts
    let init_pts = filter (\x -> let (_,q) = x in q) (zip pavgs sel)
    let init_justpts = map (\x -> let (p,_) = x in p) init_pts
    let f32z = f32.f64 0.0
    let zerod = map2 (\s i -> if i then s else {x=f32z,y=f32z,z=f32z}) pavgs (rotate 1 is_change)
    let filled = reverse (scan fill {x=f32z,y=f32z,z=f32z} (reverse zerod))
    let davgd = map2 (\pt mu -> pt v3.-mu) sorted_pts filled
    let ops = map(\pt -> mat_to_m33 (v3_outerproduct (pt,pt))) davgd
    let covsums = sgm_scan_addm ops is_change
    let covmats = map2 (\m3 ct -> (m33div m3 (f32.i32 ct))) covsums cts
    let init_covs = filter (\x -> let (_,q) = x in q) (zip covmats (rotate 1 sel))
    let init_justcovs = map (\x -> let (p,_) = x in p) init_covs
    let covsums_r = map (\x -> m33_to_mat x) init_justcovs
    in
    (init init_justpts,init covsums_r)

type chol = {l11:ftype, l21: ftype, l31: ftype, l22: ftype, l23: ftype, l33: ftype}

entry cholfact (m: [3][3]ftype) : chol =
    let l11 = f32.sqrt(m[0,0])
    let l21 = m[1,0]/l11
    let l31 = m[2,0]/l11
    let l22 = f32.sqrt(m[1,1] - l21*l21)
    let l23 = (m[2,1] - l31*l21)/l22
    let l33 = f32.sqrt(m[2,2] - (l31*l31 + l23*l23))
    in
    {l11=l11,l21=l21,l31=l31,l22=l22,l23=l23,l33=l33}

entry ch_isinvalid (c: chol) : bool =
    f32.isnan(c.l11) || f32.isnan(c.l21) || f32.isnan(c.l31) || f32.isnan(c.l22) || f32.isnan(c.l23) || f32.isnan(c.l33) || c.l11 <= (f32.f64 1e-13) || c.l22 <= (f32.f64 1e-13) || c.l33 <= (f32.f64 1e-13)

let fsubs (c: chol) (v: v3.vector) : v3.vector =
    let x = v.x/c.l11
    let y = (v.y - c.l21*x)/c.l22
    let z = (v.z - c.l31*x - c.l23*y)/c.l33
    in
    {x=x,y=y,z=z}

let bsubs (c: chol) (v: v3.vector) : v3.vector =
    let z = v.z/c.l33
    let y = (v.y - c.l23*z)/c.l22
    let x = (v.x - c.l31*z - c.l21*y)/c.l11
    in
    {x=x,y=y,z=z}

let full (c: chol) : [3][3]ftype =
    [[c.l11*c.l11,                 c.l21*c.l11,               c.l31*c.l11],
     [c.l21*c.l11, c.l21*c.l21 + c.l22 + c.l22, c.l31*c.l21 + c.l23*c.l22],
     [c.l31*c.l11, c.l31*c.l21 + c.l23*c.l22  , c.l31*c.l31 + c.l23*c.l23 + c.l33*c.l33]]


type wg = {mu: v3.vector, w: ftype, ch: chol}

entry chsolve (c: wg, v: v3.vector) : v3.vector =
    bsubs c.ch (fsubs c.ch v)


entry initialize_weighted_gaussians [n] (points: [n]v3.vector, res: ftype) : []wg =
    let (pts, covs) = voxel_grid_downsample (points,res)
    let ww = f32.f64 (1.0/(f64.i32 (length pts)))
    let wgs = map2 (\pt cov -> {mu=pt, w= ww, ch=cholfact cov}) pts covs in
    wgs

let det (c: chol) : ftype =
    let ptr = c.l11*c.l22*c.l33
    in
    ptr*ptr

let det33 (m: [3][3]ftype) : ftype =
    let a = m[0,0]
    let b = m[0,1]
    let c = m[0,2]
    let d = m[1,0]
    let e = m[1,1]
    let f = m[1,2]
    let g = m[2,0]
    let h = m[2,1]
    let i = m[2,2]
    in
    a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h - e*g)

entry pdf (w: wg, p: v3.vector) : ftype =
    let demeanp = p v3.- w.mu
    let mmul = bsubs w.ch (fsubs w.ch demeanp)
    let top = f32.exp ((f32.f64 (-0.5)) * (v3.dot demeanp mmul))
    let two = f32.f64 2.0
    let mul = (two*f32.pi)*(two*f32.pi)*(two*f32.pi)
    let bot = f32.sqrt(mul*(det w.ch))
    in
    top/bot

let gamma [m] (wgs: [m]wg) (point: v3.vector) (wga: wg) : ftype =
    let bottom = reduce (+) (f32.f64 0.0) (map (\wgt -> wgt.w*pdf(wgt, point)) wgs)
    let top = wga.w*pdf(wga, point)
    in
    top/bottom

let wpdf (w: wg, pt: v3.vector): ftype =
    if w.w <= (f32.f64 1e-13) then
        (f32.f64 0.0)
    else
        let pd = pdf(w, pt)
        in
        if pd <= (f32.f64 1e-13) then
            (f32.f64 0.0)
        else
            w.w*pd

entry log_likelihood [n] [m] (points: [n]v3.vector, wgs: [m]wg) : ftype =
    let logs = map (\p ->
       let wpdfs =  map (\x -> wpdf (x,p)) wgs
       let spdfs = reduce (+) (f32.f64 0.0) wpdfs
       in
       if spdfs <= (f32.f64 1e-13) then (f32.f64 0.0) else f32.log spdfs
    ) points
    in
    reduce (+) (f32.f64 0.0) logs

let innovate [n] [m] (points: [n]v3.vector, wgs: [m]wg, wgt: wg) : wg =
    let gammaf = gamma wgs
    let gvals = map (\pt ->
        gammaf pt wgt
    ) points
    let gpvals = map2 (\pt g ->
        v3.scale g pt
    ) points gvals
    let f32z = f32.f64 0.0
    let gsum = reduce (+) f32z gvals
    --let gsum = if (gsuma <= (f32.f64 1e-13) || (f32.isnan gsuma)) then (f32.f64 0.0) else gsuma
    let new_weight = gsum/f32.i32(n)
    let gpsum = reduce (v3.+) {x=f32z,y=f32z,z=f32z} gpvals
    let new_mu = if gsum <= (f32.f64 1e-13) then {x=f32z,y=f32z,z=f32z} else (v3.scale ((f32.f64 1.0)/gsum) gpsum)
    let zmat = mat_to_m33 (replicate 3 (replicate 3 f32z))
    let nctop = map2 (\pt g ->
        let cp = pt v3.- new_mu
        let op = v3_outerproduct (cp,cp)
        let mm = mat_to_m33 op
        let sc = m33mul mm g
        in
        sc
        ) points gvals
    let nctops = reduce m33add zmat nctop
    let new_cov = if gsum <= (f32.f64 1e-13) then zmat else (m33div nctops gsum)
    let new_cov_mat = m33_to_mat new_cov
    in
    {mu=new_mu, w=new_weight,ch=cholfact new_cov_mat}

entry em_fit_hgmm [n] [m] (points: [n]v3.vector, wgs: [m]wg) : []wg =
    let wgnew = map (\x -> innovate (points,wgs,x)) wgs
    in
    filter (\x -> if (ch_isinvalid x.ch) then false else true) wgnew

let compute_assignments [n] [m] (points: [n]v3.vector) (wgs: [m] wg) : [m][n]ftype =
    let pprobs = map (\wg ->
        map (\pt ->
            wpdf (wg,pt)
        ) points
    ) wgs
    let wsums = map (\w ->
        reduce (+) (f32.f64 0.0) w
    ) pprobs
    let pnorm = map (\(row,su) ->
        map (\asdf ->
            if asdf <= (f32.f64 1e-13) then
                (f32.f64 0.0)
            else
                if su <= (f32.f64 1e-13) then
                    (f32.f64 0.0)
                else
                    asdf/su) row
    ) (zip pprobs wsums)
    in
    pnorm

let update [n] [m] (points: [n]v3.vector) (atp: [m][n]ftype): [m]wg =
    let gsum = map (\r -> reduce (+) (f32.f64 0.0) r) atp
    let pis = map (\uww -> ((f32.f64 1.0)/(f32.i32 n))*uww) gsum
    let mus_top = map (\r ->
        let m = map (\(gamma,pt) -> v3.scale gamma pt) (zip r points)
        let v3z = {x=f32.f64 0.0,y= f32.f64 0.0,z= f32.f64 0.0}
        in
        reduce (v3.+) v3z m
    ) atp
    let mus = map (\(m,uww) -> v3.scale ((f32.f64 1.0)/uww) m) (zip mus_top gsum)
    let covs = map (\(row,bot,mu) ->
        let cvs = map (\(gamma,pt) ->
            let demean = pt v3.- mu
            let op = v3_outerproduct (demean, demean)
            let mm = mat_to_m33 op
            in
            m33mul mm gamma
        ) (zip row points)
        let f32z = f32.f64 0.0
        let m33z = mat_to_m33 (replicate 3 (replicate 3 f32z))
        let cr = reduce m33add m33z cvs
        in
        m33_to_mat (m33div cr bot)
    ) (zip3 atp gsum mus)
    in
    map (\(pi, mu, cov) -> {mu=mu, w=pi,ch=cholfact cov}) (zip3 pis mus covs)

let update2 [n] [m] (points: [n]v3.vector) (atp: [m][n]ftype): [m]wg =
    let q = map (\atpi ->
        let opr = map (\(gamma,pt) ->
            let op = v3_outerproduct (pt, pt)
            let mm = mat_to_m33 op
            let sc = m33mul mm gamma
            let gsc = v3.scale gamma pt
            in
            (sc, gsc)
        ) (zip atpi points)
        let (opa, pa) = unzip opr
        let f32z = f32.f64 0.0
        let v3z = {x=f32z,y=f32z,z=f32z}
        let m33z = mat_to_m33 (replicate 3 (replicate 3 f32z))
        let cr = reduce m33add m33z opa
        let pp = reduce (v3.+) v3z pa
        let gg = reduce (+) f32z atpi
        in
        (cr,pp,gg)
    ) atp
    let (ops,ps,gs) = unzip3 q
    let mus = map (\(mua, gs) ->
        v3.scale ((f32.f64 1.0)/gs) mua
    ) (zip ps gs)
    let csubs = map (\mu ->
        mat_to_m33 (v3_outerproduct (mu,mu))
    ) mus
    let covs = map (\(psu,musu,gam) ->
        let a = m33div psu gam
        --let b = m33div musu gam
        in
        m33sub a musu
    ) (zip3 ops csubs gs)
    let pis = map (\gamma ->
        gamma/(f32.i32 n)
    ) gs
    in
    map (\(pi, mu, cov) -> {mu=mu, w=pi,ch=cholfact (m33_to_mat cov)}) (zip3 pis mus covs)

let mu_isinvalid (v: v3.vector) : bool =
    f32.isnan(v.x) || f32.isnan(v.y) || f32.isnan(v.z)

let w_isinvalid (w: wg) : bool =
    ch_isinvalid w.ch || mu_isinvalid w.mu || f32.isnan w.w

entry em_fit_hgmm2 [n] [m] (points: [n]v3.vector, wgs: [m]wg) : []wg =
    let assgn = compute_assignments points wgs
    let uw = update2 points assgn
    in
    uw

entry filter_bad [m] (wgs: [m]wg) : []wg =
    filter (\x -> if (w_isinvalid x) then false else true) wgs
