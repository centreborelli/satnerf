#!/usr/bin/python

"""
A DTM raster can be represented by triangle meshes by finding a set of non-overlapping triangles that covers the entire mesh and approximates the elevation field. 
There are two different types of triangle meshes that can be used for this purpose:

(1) A triangulated regular network (TRN), in which every pixel of the raster is represented by a vertex, and all triangles have the same size and shape. 
    All the original information of the DTM raster is present in the TRN, but the memory required for storing the mesh is typically quite high.
(2) A triangulated irregular network (TIN), in which there are fewer vertices than raster pixels and the triangles have different shapes and sizes. 
    The vertices and the triangulation are chosen in such a way that the resulting surface approximates the original DTM raster up to a specified error.
    This typically results in much smaller files, since plane or nearly plane areas can be represented using only a couple of vertices.

In most applications, if you need to deal with elevation meshes, you'd go with a TIN since throwing out redundant information allows for more efficient computations.
However, creating TINs from rasters isn't straightforward, since there are multiple triangulations that approximate a grid with similar error, but using different vertex sets.

This script creates a mesh from a raster via a triangulated regular network (TRN)
source: https://gis.stackexchange.com/questions/121561/seeking-tool-to-generate-mesh-from-dtm
"""



import sys
import numpy as np
from osgeo import gdal

def write_ply(filename, coordinates, triangles, binary=True):
    template = "ply\n"
    if binary:
        template += "format binary_" + sys.byteorder + "_endian 1.0\n"
    else:
        template += "format ascii 1.0\n"
    template += """element vertex {nvertices:n}
property float x
property float y
property float z
element face {nfaces:n}
property list int int vertex_index
end_header
"""

    context = {
     "nvertices": len(coordinates),
     "nfaces": len(triangles)
    }

    if binary:
        with  open(filename,'wb') as outfile:
            outfile.write(template.format(**context))
            coordinates = np.array(coordinates, dtype="float32")
            coordinates.tofile(outfile)

            triangles = np.hstack((np.ones([len(triangles),1], dtype="int") * 3,
                triangles))
            triangles = np.array(triangles, dtype="int32")
            triangles.tofile(outfile)
    else:
        with  open(filename,'w') as outfile:
            outfile.write(template.format(**context))
            np.savetxt(outfile, coordinates, fmt="%.3f")
            np.savetxt(outfile, triangles, fmt="3 %i %i %i")

def readraster(filename):
    raster = gdal.Open(filename)
    return raster


def createvertexarray(raster):
    transform = raster.GetGeoTransform()
    width = raster.RasterXSize
    height = raster.RasterYSize
    x = np.arange(0, width) * transform[1] + transform[0]
    y = np.arange(0, height) * transform[5] + transform[3]
    xx, yy = np.meshgrid(x, y)
    zz = raster.ReadAsArray()
    vertices = np.vstack((xx,yy,zz)).reshape([3, -1]).transpose()
    return vertices


def createindexarray(raster):
    width = raster.RasterXSize
    height = raster.RasterYSize

    ai = np.arange(0, width - 1)
    aj = np.arange(0, height - 1)
    aii, ajj = np.meshgrid(ai, aj)
    a = aii + ajj * width
    a = a.flatten()

    tria = np.vstack((a, a + width, a + width + 1, a, a + width + 1, a + 1))
    tria = np.transpose(tria).reshape([-1, 3])
    return tria


def main(argv):
    inputfile = argv[0]
    outputfile = argv[1]

    raster = readraster(inputfile)
    vertices = createvertexarray(raster)
    triangles = createindexarray(raster)

    write_ply(outputfile, vertices, triangles, binary=False)

if __name__ == "__main__":

    # usage example: $ python geotiff2mesh.py file.tif file.ply
    # input geotiff can contain NaN values
    main(sys.argv[1:])
