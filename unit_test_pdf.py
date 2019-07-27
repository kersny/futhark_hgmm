import hgmm
import cv2
import numpy
import unittest
import scipy.stats

numpy.set_printoptions(threshold=numpy.nan)

PATH="/home/kersny/foo/home_2018_05_10/projects/halide_stuff/data/livingRoom0/"
#PATH = "livingRoom0/"

# pose is tf to world
poses = numpy.loadtxt(PATH+"livingRoom0n.gt.sim")


def get_pose(index):
    global poses
    return [poses[3*index:3*index+3, 0:3], poses[3*index:3*index+3, 3]]


def read_image_depth(index):
    global PATH
    image = cv2.imread(PATH+"/rgb/{}.png".format(index), 0)
    depth = cv2.imread(PATH+"/depth/{}.png".format(index), -1)
    return image, depth


FSTR = """{ch:{l11:f32, l21:f32, l22:f32, l23:f32, l31:f32, l33:f32}, mu:{x:f32, y:f32,
                                                                z:f32}, w:f32}"""

DEPTH_TO_POINTCLOUD_ARG = """([][]u16, {cx: f32, cy: f32, fx: f32, fy: f32}, {R: [][]f32, t: {x: f32, y: f32,
                                                                 z: f32}})"""
INITIALIZE_WEIGHTED_GAUSSIANS_ARG = """([]{x: f32, y: f32, z: f32}, f32)"""


class TestPdf(unittest.TestCase):
    def test_pdf(self):
        global FSTR
        hgmmi = hgmm.hgmm()

        intr = hgmm.opaque('{cx:f32, cy:f32, fx:f32, fy:f32}', 481.2,
                           -480.0, 391.5, 239.5)
        idx = 0
        im, depth = read_image_depth(idx)
        pose = get_pose(idx)
        poseo = hgmm.opaque('{R:[][]f32, t:{x:f32, y:f32, z:f32}}',
                            pose[0].astype(numpy.float32), pose[1][0],
                            pose[1][1], pose[1][2])

        dtp_arg = hgmm.opaque(DEPTH_TO_POINTCLOUD_ARG,
                depth.astype(numpy.int16), 481.2,
                           -480.0, 391.5, 239.5, pose[0].astype(numpy.float32), pose[1][0],
                            pose[1][1], pose[1][2])
        allpts = hgmmi.depth_to_pointcloud(dtp_arg)
        iwg_arg = hgmm.opaque(INITIALIZE_WEIGHTED_GAUSSIANS_ARG, allpts.data[0], allpts.data[1], allpts.data[2], 0.5)
        wgs = hgmmi.initialize_weighted_gaussians(iwg_arg)
        l11 = wgs.data[0]
        l21 = wgs.data[1]
        l22 = wgs.data[2]
        l23 = wgs.data[3]
        l31 = wgs.data[4]
        l33 = wgs.data[5]
        mux = wgs.data[6]
        muy = wgs.data[7]
        muz = wgs.data[8]
        w = wgs.data[9]

        L = numpy.asmatrix([[l11[0].get(), l21[0].get(), l31[0].get()],
                           [0.0, 	   l22[0].get(), l23[0].get()],
                           [0.0,		    0.0, l33[0].get()]])

        cov = L.T*L
        mu = numpy.asarray([mux[0].get(),
                             muy[0].get(),
                             muz[0].get()])

        wg1 = hgmm.opaque(FSTR, l11[0].get(), l21[0].get(), l22[0].get(),
                          l23[0].get(), l31[0].get(), l33[0].get(),
                          mux[0].get(), muy[0].get(), muz[0].get(),
                          w[0].get())
        gt = scipy.stats.multivariate_normal(mean=mu, cov=cov)
        ct = 0
        for i in range(len(allpts.data[0])):
            ptn = numpy.asarray([allpts.data[0][i].get(), allpts.data[1][i].get(), allpts.data[2][i].get()])
            pti = hgmm.opaque('{x:f32, y:f32, z:f32}', allpts.data[0][i].get(), allpts.data[1][i].get(), allpts.data[2][i].get())
            # pdf_gt = numpy.linalg.solve(cov, ptn)
            # pdf_est = hgmmi.chsolve(wg1, pti)
            # pdf_est_v = numpy.asarray([pdf_est.data[0], pdf_est.data[1], pdf_est.data[2]])
            pdf_gt = gt.pdf(ptn)
            pdf_est_v = hgmmi.pdf(wg1, pti)
            err = numpy.linalg.norm(pdf_gt-pdf_est_v)
            if err > 1e-8:
                ct += 1
        print(ct,",",len(allpts.data[0]))


if __name__ == '__main__':
    unittest.main()
