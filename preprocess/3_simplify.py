import os
import argparse
import ntpath
import common
import pdb
import open3d as o3d
import numpy as np

class Simplification:
    """
    Perform simplification of watertight meshes.
    """

    def __init__(self):
        """
        Constructor.
        """

        parser = self.get_parser()
        self.options = parser.parse_args()
        self.simplification_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'simplification.mlx')

    def get_parser(self):
        """
        Get parser of tool.

        :return: parser
        """

        parser = argparse.ArgumentParser(description='Scale a set of meshes stored as OFF files.')
        parser.add_argument('--in_dir', type=str, help='Path to input directory.')
        parser.add_argument('--out_dir', type=str, help='Path to output directory; files within are overwritten!')

        return parser

    def read_directory(self, directory):
        """
        Read directory.

        :param directory: path to directory
        :return: list of files
        """

        files = []
        for filename in os.listdir(directory):
            files.append(os.path.normpath(os.path.join(directory, filename)))

        return files

    def run(self):
        """
        Run simplification.
        """

        if not os.path.exists(self.options.in_dir):
            return
        common.makedir(self.options.out_dir)
        files = self.read_directory(self.options.in_dir)

        # count number of faces
        num_faces = []

        for filepath in files:
            mesh = o3d.io.read_triangle_mesh(filepath)
            faces = np.asarray(mesh.triangles).shape[0]
            num_faces.append(faces)

        num_faces = np.array(num_faces)
        total_faces = np.sum(num_faces)
        num_faces = np.around(2500 * (num_faces / (total_faces+0.0))).astype(int) # total 2500 faces


        for idx, filepath in enumerate(files):

            # write new simply mlx file
            with open(os.path.join(self.options.out_dir,'tmp.mlx'), 'w') as out_file:
                with open(self.simplification_script, 'r') as in_file:
                    Lines = in_file.readlines()
                    for count, line in enumerate(Lines):
                        # modify target face number according to ratio
                        if count == 3:
                            front = line[:51]
                            back = line[57:]
                            line = front+"\""+str(num_faces[idx])+"\""+back
                        out_file.write(line)
            os.system('meshlabserver -i %s -o %s -s %s' % (
                filepath,
                os.path.join(self.options.out_dir, ntpath.basename(filepath)),
                os.path.join(self.options.out_dir,'tmp.mlx')
            ))

if __name__ == '__main__':
    app = Simplification()
    app.run()
