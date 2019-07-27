import hgmm
import cv2
import numpy
import time
import os

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

hgmmi = hgmm.hgmm()

intr = hgmm.opaque('{cx:f32, cy:f32, fx:f32, fy:f32}', 481.2,
                   -480.0, 391.5, 239.5)
idx = 0
im, depth = read_image_depth(idx)
pose = get_pose(idx)
poseo = hgmm.opaque('{R:[][]f32, t:{x:f32, y:f32, z:f32}}',
                   pose[0].astype(numpy.float32), pose[1][0], pose[1][1], pose[1][2])

allpts = hgmmi.depth_to_pointcloud(depth, intr, poseo)
wgsa = hgmmi.initialize_weighted_gaussians(allpts, 0.5)
print(len(wgsa.data[0]))
prev_ll = hgmmi.log_likelihood(allpts, wgsa)
print(prev_ll)
wgs = hgmmi.filter_bad(wgsa)
print(len(wgs.data[0]))
fstr = """{ch:{l11:f32, l21:f32, l22:f32, l23:f32, l31:f32, l33:f32}, mu:{x:f32, y:f32,
                                                                z:f32}, w:f32}"""
#
#pdf = hgmmi.pdf(wg1, pt1)
#print(pdf)
prev_ll = hgmmi.log_likelihood(allpts, wgs)
print(prev_ll)
for i in range(100):
    tstart = time.time()
    wgsa = hgmmi.em_fit_hgmm2(allpts, wgs)
    tmid1 = time.time()
    wgs = hgmmi.filter_bad(wgsa)
    tmid2 = time.time()
    new_ll = hgmmi.log_likelihood(allpts, wgs)
    #if not numpy.isfinite(new_ll):
    #    for d in wgs.data:
    #        print(d)
    delta_ll = new_ll - prev_ll
    tend = time.time()
    dtfit = tmid1 - tstart
    dtfilt = tmid2 - tmid1
    dtll = tend - tmid2
    print(len(wgs.data[0]), new_ll, delta_ll, dtfit, dtfilt, dtll)
    prev_ll = new_ll


# TODO: implement explicit assignment based innovation & see if it's faster
# also better nan handling pretty much everywhere
# also implement averaging as reduction (no scan sum hopefully) for more precision
