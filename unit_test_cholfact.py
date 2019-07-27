import hgmm
import cv2
import numpy
import unittest

numpy.set_printoptions(threshold=numpy.nan)

PATH = "livingRoom0/"

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


class TestCholesky(unittest.TestCase):
    def test_cholesky(self):
        hgmmi = hgmm.hgmm()

        intr = hgmm.opaque('{cx:f32, cy:f32, fx:f32, fy:f32}', 481.2,
                           -480.0, 391.5, 239.5)
        idx = 0
        im, depth = read_image_depth(idx)
        pose = get_pose(idx)
        poseo = hgmm.opaque('{R:[][]f32, t:{x:f32, y:f32, z:f32}}',
                            pose[0].astype(numpy.float32), pose[1][0],
                            pose[1][1], pose[1][2])

        allpts = hgmmi.depth_to_pointcloud(depth, intr, poseo)
        (pts, covs) = hgmmi.voxel_grid_downsample(allpts, 0.1)
        print(len(pts.data[0]))

        covs_n = covs.get()

        #  {l11:f32, l21:f32, l22:f32, l23:f32, l31:f32, l33:f32}
        nbad = 0
        nverybad = 0
        ngood = 0
        nvgood = 0
        for i in range(covs_n.shape[0]):
            ch_fut_s = hgmmi.cholfact(covs_n[i])
            l11 = ch_fut_s.data[0]
            l21 = ch_fut_s.data[1]
            l22 = ch_fut_s.data[2]
            l23 = ch_fut_s.data[3]
            l31 = ch_fut_s.data[4]
            l33 = ch_fut_s.data[5]
            ch_fut = numpy.asarray([[l11, 0.0, 0.0],
                                    [l21, l22, 0.0],
                                    [l31, l23, l33]])
            try:
                ch_np = numpy.linalg.cholesky(covs_n[i])
                if numpy.allclose(ch_np, ch_fut, rtol=1e-2, atol=1e-7):
                    ngood += 1
                    # print("pass: close")
                    pass
                else:
                    nbad += 1
                    # print("fail: notclose")
                    # print(ch_fut)
                    # print(numpy.abs(ch_fut-ch_np))
            except numpy.linalg.linalg.LinAlgError:
                if not hgmmi.ch_isinvalid(ch_fut_s):
                    nverybad += 1
                    # print("fail: cholesky + valid")
                else:
                    nvgood += 1
                    # print("pass: cholesky + invalid")
                    pass
        total = nbad+nvgood+ngood+nverybad
        pct_bad = float(nbad)/float(total)
        pct_bad_errd = float(nvgood)/float(total)
        pct_verybad = float(nverybad)/float(total)
        print(total, pct_bad, pct_verybad, pct_bad_errd)
        self.assertTrue(pct_verybad <= 0.01)
        self.assertTrue(pct_bad <= 0.07)


if __name__ == '__main__':
    unittest.main()
