"""Projective Homography and Panorama Solution."""
import matplotlib.pyplot as plt
import numpy
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata


PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """

        match_p_src = match_p_src.astype(int)
        match_p_dst = match_p_dst.astype(int)
        A = np.zeros((2*match_p_src.shape[1],9))
        for i in range(match_p_src.shape[1]): #creating 2 rows for each point in matches_mat
            A[2*i] = [match_p_src[0][i],match_p_src[1][i],1,0,0,0,-match_p_dst[0][i]*match_p_src[0][i],-match_p_dst[0][i]*match_p_src[1][i],-match_p_dst[0][i]]
            A[2*i+1] = [0,0,0,match_p_src[0][i], match_p_src[1][i], 1 , -match_p_dst[1][i] * match_p_src[0][i], -match_p_dst[1][i] * match_p_src[1][i], -match_p_dst[1][i]]
        U,S,V = np.linalg.svd(A)
        H = np.reshape(V[-1],(3,3))
        return H
        ##Homography =np.array[2N][9]
        #print(match_p_src.shape[1])
        # for i in range(match_p_src.shape[1]):
        #     a = []
        #     b = []
        #     x_1 = match_p_src[0][i]
        #     y_1 = match_p_src[1][i]
        #     x_2 = match_p_dst[0][i]
        #     y_2 = match_p_dst[1][i]
        #     a.append(x_1)
        #     a.append(y_1)
        #     a.append(1)
        #     a.append(0)
        #     a.append(0)
        #     a.append(0)
        #     a.append(-1*x_1*x_2)
        #     a.append(-1*x_2*y_2)
        #     a.append(-1*x_2)
        #     b.append(0)
        #     b.append(0)
        #     b.append(0)
        #     b.append(x_1)
        #     b.append(x_2)
        #     b.append(1)
        #     b.append(-1*x_1*y_2)
        #     b.append(-1*y_1*y_2)
        #     b.append(-1*y_2)
        #     a = np.array(a)
        #     b = np.array(b)
        #     c = np.vstack((a, b))
        #     if (i==0):
        #         A=c
        #     else:
        #         A = np.vstack((A, c))
        #
        # u, s, vh = np.linalg.svd(A)
        # H = np.reshape(vh[-1],(3,3))
        # #h_vector = vh[:,-1]
        # #H = np.reshape(h_vector, (3,3))
        # return H

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:

        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """

        """INSERT YOUR CODE HERE"""
        new_image = np.zeros(dst_image_shape)

        for u in range(src_image.shape[0]):
            for v in range(src_image.shape[1]):
                u_new = (np.dot(homography[0,:] ,np.array([u,v,1])))/ (np.dot(homography[2,:],np.array([u,v,1])))
                u_new_floor = u_new.astype(np.uint)

                v_new = (np.dot(homography[1,:] ,np.array([u,v,1])))/ (np.dot(homography[2,:],np.array([u,v,1])))
                v_new_floor = v_new.astype(np.uint)

                if (u_new_floor < dst_image_shape[0]) and (v_new_floor < dst_image_shape[1]): #check going out of bounds
                    new_image[u_new_floor, v_new_floor,:] = src_image[u,v, :]

        return new_image


    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        """INSERT YOUR CODE HERE"""
        # # x = np.arange(src_image.shape[0])
        # # y = np.arange(src_image.shape[1])
        # # xy = np.meshgrid(x, y)
        # #
        # #
        # # xyf = np.zeros((3, (src_image.shape[0]*src_image.shape[1])))
        # #
        # # xyf[0, :] = np.array(xy[0].flatten())
        # # xyf[1, :] = np.array(xy[1].flatten())
        # # xyf[2, :] = np.ones(src_image.shape[0]*src_image.shape[1])
        # #
        # indices = np.indices((src_image.shape[0], src_image.shape[1]))
        # # Add row of 1's
        # indices_hom = np.vstack((indices, np.ones((1, indices.shape[1], indices.shape[2])))).reshape(3, -1)
        # #
        # #
        # #xyf_t = np.matmul(homography, xyf)
        # xyf_t = np.dot(homography, indices_hom)
        # xyf_t[0, :] = xyf_t[0, :] / xyf_t[2, :]
        # xyf_t[1, :] = xyf_t[1, :] / xyf_t[2, :]
        # xyf_t[2, :] = xyf_t[2, :] / xyf_t[2, :]
        #
        #
        # xyf_t = np.round(xyf_t).astype(int)
        # #
        # # # Round the coordinates. In effect, this is
        # # # nearest-neighbour interpolation
        # xyf_t = np.around(xyf_t).astype(np.int32)
        # # #xyf_t = np.around(np.dot(homography, xyf)).astype(np.int16)
        # #
        # # # Reshape the arrays back to be in the same format as
        # # # xy
        # xy_t = [xyf_t[0, :].reshape(xy[0].shape),
        #          xyf_t[1, :].reshape(xy[1].shape)]
        # x_new = np.array(xyf_t[0, :])
        # y_new = np.array(xyf_t[1, :])
        # #
        # new_image = np.zeros(dst_image_shape)
        # condition1 = (x_new <= dst_image_shape[0])
        # condition2 = (y_new <= dst_image_shape[1])
        # new_image[np.where(xy_t[0] < dst_image_shape[0]),np.where(xy_t[1] < dst_image_shape[1])] = src_image[xy[0],xy[1]]
        # #
        # new_image[np.select(condition1,x_new),np.select(condition2,y_new),:] = src_image[xy[0],xy[1],:]
        # new_image[condition1,condition2,:] = src_image[xy[0],xy[1],:]
        # return new_image
        # #pass

        # Yarden's code:

        # Create meshgrid
        meshgrid = np.indices((src_image.shape[0], src_image.shape[1]))
        # Create homogoneous coordinates matrix
        hom_matrix = np.vstack((meshgrid, np.ones((1, meshgrid.shape[1], meshgrid.shape[2])))).reshape(3, -1)
        # Matrix multiplication
        trans_matrix = np.dot(homography,hom_matrix)
        # Normalize to get real values
        x_new = np.round(trans_matrix[0] / trans_matrix[2]).astype(np.int).reshape(src_image.shape[0],src_image.shape[1])
        y_new = np.round(trans_matrix[1] / trans_matrix[2]).astype(np.int).reshape(src_image.shape[0],src_image.shape[1])
        # Make sure the pixels in the correct range
        x_new = np.clip(x_new, 0, dst_image_shape[0] - 1)
        y_new = np.clip(y_new, 0, dst_image_shape[1] - 1)
        new_image = np.zeros(dst_image_shape)
        new_image[x_new, y_new] = src_image[meshgrid[0], meshgrid[1]]
        return new_image

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        # return fit_percent, dist_mse
        """INSERT YOUR CODE HERE"""

        xy_src = np.vstack([match_p_src,np.ones(match_p_src.shape[1])])
        xy_dst = np.vstack([match_p_dst,np.ones(match_p_dst.shape[1])])
        transformed_xy_src = np.dot(homography,xy_src)
        # Normalize
        transformed_xy_src = transformed_xy_src / transformed_xy_src[2,:]
        error = transformed_xy_src - xy_dst

        abs_value_error = np.array([np.linalg.norm(error[:2, i]) for i in range(error.shape[1])])  #not using the 1s in homogoneous coordinates
        fit_points = [i for i in range(len(abs_value_error)) if abs_value_error[i] < max_err]
        fit_percent = len(fit_points) / match_p_src.shape[1]
        if not fit_percent:
            dist_mse = 10**9
        else:
            dist_mse = np.mean(abs_value_error[fit_points])
        return fit_percent, dist_mse

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        # return mp_src_meets_model, mp_dst_meets_model
        """INSERT YOUR CODE HERE"""
        xy_src = np.vstack([match_p_src, np.ones(match_p_src.shape[1])])
        xy_dst = np.vstack([match_p_dst, np.ones(match_p_dst.shape[1])])
        transformed_xy_src = np.dot(homography, xy_src)
        # Normalize
        transformed_xy_src = transformed_xy_src / transformed_xy_src[2, :]
        error = transformed_xy_src - xy_dst

        abs_value_error = np.array([np.linalg.norm(error[:2, i]) for i in
                                    range(error.shape[1])])  # not using the 1s in homogoneous coordinates
        fit_points = [i for i in range(len(abs_value_error)) if abs_value_error[i] < max_err]
        return (match_p_src[:,fit_points],match_p_dst[:,fit_points])

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # use class notations:
        w = inliers_percent
        t = max_err
        # p = parameter determining the probability of the algorithm to
        # succeed
        p = 0.99
        # the minimal probability of points which meets with the model
        d = 0.5
        # number of points sufficient to compute the model
        n = 4
        # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
        # return homography
        """INSERT YOUR CODE HERE"""
        number_of_points = match_p_src.shape[1] #total number of points
        error = np.inf
        H = []
        for _ in range(k): # LOOP for k iterations
            points_indices = np.random.randint(1,number_of_points,n)
            src_points = match_p_src[:,points_indices]
            dst_points = match_p_dst[:,points_indices]
            curr_h = self.compute_homography_naive(src_points,dst_points)
            fit_percent, error = self.test_homography(curr_h,match_p_src,match_p_dst,t)
            if fit_percent > d:
                meet_src_points,meet_dst_points = self.meet_the_model_points(curr_h,match_p_src,match_p_dst,t)
                recomputed_h = self.compute_homography_naive(meet_src_points,meet_dst_points)
                new_fit_percent, new_error = self.test_homography(recomputed_h,match_p_src,match_p_dst,t)
                if new_error < error:
                    H = recomputed_h
        if H == []:
            print('THe algorithm did not manage to find a good result')
        return H

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """

        # return backward_warp
        """INSERT YOUR CODE HERE"""
        # Create meshgrid
        dst_meshgrid = np.indices((dst_image_shape[0], dst_image_shape[1]))
        # Create homogoneous coordinates matrix
        hom_matrix = np.vstack((dst_meshgrid, np.ones((1, dst_meshgrid.shape[1], dst_meshgrid.shape[2])))).reshape(3, -1)
        # Matrix multiplication
        trans_matrix = np.dot(backward_projective_homography, hom_matrix)

        trans_matrix[0] = trans_matrix[0] / trans_matrix[2]
        trans_matrix[1] = trans_matrix[1] / trans_matrix[2]
        trans_matrix[2] = trans_matrix[2] / trans_matrix[2]

        dst_x = trans_matrix[0,:].reshape(dst_image_shape[0],dst_image_shape[1])
        dst_y = trans_matrix[1,:].reshape(dst_image_shape[0],dst_image_shape[1])

        # x_new = np.round(trans_matrix[0] / trans_matrix[2]).astype(np.int).reshape(src_image.shape[0],
        #                                                                            src_image.shape[1])
        # y_new = np.round(trans_matrix[1] / trans_matrix[2]).astype(np.int).reshape(src_image.shape[0],
        #                                                                            src_image.shape[1])

        # Create meshgrid of source image coordinates
        source_meshgrid = np.indices((src_image.shape[0], src_image.shape[1]))
        # source_matrix = np.vstack((source_meshgrid, np.ones((1,src_image.shape[0],src_image.shape[1])))).reshape(3,-1)

        src_x = source_meshgrid[0].reshape(src_image.shape[0],src_image.shape[1])
        src_y = source_meshgrid[1].reshape(src_image.shape[0],src_image.shape[1])

        result = scipy.interpolate.griddata((src_x,src_y),src_image[src_x,src_y,:],(dst_x,dst_y),method='cubic')
        return result


    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        # return final_homography
        """INSERT YOUR CODE HERE"""
        pass

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        # return np.clip(img_panorama, 0, 255).astype(np.uint8)
        """INSERT YOUR CODE HERE"""
        pass


if __name__ == '__main__':
    import scipy.io
    matches = scipy.io.loadmat('matches.mat')
    source = plt.imread('src.jpg')
    # dst = plt.imread('dst.jpg')
    # plt.imshow(source)
    # plt.scatter(matches['match_p_src'][0],matches['match_p_src'][1],c='red')
    solution = Solution()
    # H = solution.compute_homography_naive(matches['match_p_src'],matches['match_p_dst'])
    H = solution.compute_homography(matches['match_p_src'],matches['match_p_dst'],0.8,25)
    H_inv = np.linalg.inv(H)
    backward_homography = solution.compute_backward_mapping(H_inv,source,(1088,1452,3))
    # forward_homography_slow = solution.compute_forward_homography_slow(H,source, (1088,1452,3))
    # result = forward_homography_slow.astype(np.uint8)
    # plt.imshow(result)
    # plt.show()
    # forward_homography_fast = solution.compute_forward_homography_fast(H,source, (1088,1452,3))
    # result = forward_homography_fast.astype(np.uint8)
    # plt.imshow(result)
    # plt.show()
    #print(solution.test_homography(H,matches['match_p_src'],matches['match_p_dst'],25))
    # print(solution.meet_the_model_points(H,matches['match_p_src'],matches['match_p_dst'],25))
