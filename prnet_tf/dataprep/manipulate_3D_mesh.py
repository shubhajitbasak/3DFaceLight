import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import matplotlib.tri as tri

import menpo3d.io as m3io
import menpo.io as mio
from menpo.shape import PointCloud


def getCentroids(verts):
    tris = Delaunay(verts, furthest_site=False, incremental=True)
    centroids = []
    for tri in tris.simplices:
        centroids.append([np.sum(verts[tri][:, 0]) / 3.0, np.sum(verts[tri][:, 1]) / 3.0])
    centroids = np.asarray(centroids)
    return tris, np.concatenate((verts, centroids), axis=0)


def getEquidistantPoints(p1, p2, parts):
    return zip(np.linspace(p1[0], p2[0], parts + 1),
               np.linspace(p1[1], p2[1], parts + 1))


def adjustKeypoints():
    vertices_from_blender = np.loadtxt('../data/save-img/blender/'
                                       'vertices_XXXX_sel_from_blender.txt')
    vertices_68 = np.loadtxt('../data/save-img/blender/vertices_68.txt')

    vertices_final = np.concatenate((vertices_from_blender, vertices_68), axis=0)
    indexs_final_unique = np.unique(vertices_final, axis=0)

    np.savetxt('../data/save-img/blender/'
               'vertices_XXXX_sel_to_blender.txt', indexs_final_unique)


def get_68_index_from500():
    vertices_from_blender = np.loadtxt('../data/save-img/blender/'
                                       'vertices_XXXX_sel_from_blender.txt')
    vertices_68 = np.loadtxt('../data/save-img/blender/vertices_68.txt')

    filters =[]

    for v in vertices_68:
        filters.append(np.where(vertices_from_blender == v)[0][0])

    filters = np.asarray(filters)

    np.savetxt('../data/save-img/blender/vertices_68_fil_500.txt', filters)


def applyDelaunay():
    vertices = np.loadtxt('../data/save-img/blender/Iter2/vertices_boundary_sel_from_blender.txt')
    vertices_68 = np.loadtxt('../data/save-img/blender/Iter2/vertices_68.txt')
    template = m3io.import_mesh('../data/save-img/blender/base_template.obj')

    vertices = vertices[0:].astype(int)
    vertices_68 = vertices_68[0:].astype(int)
    # del_ind = [6,7]
    # vertices = np.delete(vertices,del_ind)

    # point_cloud = PointCloud(template.points[vertices])
    vertices_all_2d = template.points[:, :2]
    vertices_3d = template.points[vertices]

    vertices_2d = vertices_3d[:, :2]

    vertices_68_3d = template.points[vertices_68]
    vertices_68_2d = vertices_68_3d[:, :2]

    fltr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,  32]  # 19, 20, 23, # 17,18,23,30,31,
    fltrArr = np.take(vertices_2d, fltr, 0)

    # verts = []
    # for _,p in enumerate(vertices_68_2d[:32]):
    #     temp = list(getEquidistantPoints(vertices_68_2d[32], p, 10))
    #     verts.extend(temp)
    #
    # verts = res = [*set(verts)]  # get unique values
    # verts = np.asarray(verts)

    # plt.plot(vertices_68_2d[:, 0], vertices_68_2d[:, 1], 'ro')
    # plt.figure()
    #
    plt.plot(fltrArr[:, 0], fltrArr[:, 1], 'ro')
    plt.figure()

    vertices_2d_new = fltrArr.copy()

    delaunay_iter = 3
    for i in range(0, delaunay_iter):
        tris, vertices_2d_new = getCentroids(vertices_2d_new)

    # *********** Calculate nearest Neighbour***********
    # ---- Calculate nearest neighbour from the new vertices in the original mesh
    ctree = cKDTree(vertices_all_2d)
    indexs = []
    for vert in vertices_2d_new:
        ds, inds = ctree.query(vert, 1)
        # print(ds, inds)
        indexs.append(inds)
    # ***********End Calculate Neighbour***********

    indexs = np.asarray(indexs)

    indexs_final = np.concatenate((vertices, vertices_68, indexs), axis=0)
    indexs_final_unique = np.unique(indexs_final, axis=0)

    # print('Total starting vertices : ', len(vertices))
    # print('Total vertices after {0} iteration of delaunay : '.format(delaunay_iter), len(vertices_68_2d_new))
    # print('Total neighourest neighbour vertices in original mesh :', len(indexs))
    # ds, inds = ctree.query(vertices_68_2d_new[0], 1)
    # np.savetxt('../data/save-img/blender/Iter2/vertices_sel_after_delaunay.txt', indexs_final_unique)

    plt.triplot(vertices_2d_new[:, 0], vertices_2d_new[:, 1], tris.simplices)
    plt.plot(vertices_2d_new[:, 0], vertices_2d_new[:, 1], 'ro')

    plt.show()


if __name__ == '__main__':
    applyDelaunay()
    # get_68_index_from500()
    # adjustKeypoints()
    # sampleMatplot()
