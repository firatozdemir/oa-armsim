# Author: Firat Ozdemir, May 2022, firat.ozdemir@datascience.ch

import numpy as np
from scipy.ndimage import gaussian_filter
import scipy.spatial.transform

class NormDist:
    def __init__(self, mu, sig, multiplier=1.):
        self.mu = mu
        self.sig = sig
        self.multiplier = multiplier
    def __call__(self, x):
        return np.exp(-(x-self.mu)**2/self.sig**2) * self.multiplier

def cyr_dist_fn(p,a1,a2):
    '''p: point of query
    a1, a2: 2 points that pass through center of the cylinder core. (x,y) from a1 can be used to shift origin'''
    d = np.sqrt((((a2[1]-a1[1])*(a1[2]-p[2])-(a2[2]-a1[2])*(a1[1]-p[1]))**2 + ((a2[2]-a1[2])*(a1[0]-p[0])-(a2[0]-a1[0])*(a1[2]-p[2]))**2 + ((a2[0]-a1[0])*(a1[1]-p[1])-(a2[1]-a1[1])*(a1[0]-p[0]))**2)/((a2[0]-a1[0])**2 + (a2[1]-a1[1])**2 + (a2[2]-a1[2])**2))
    return d

def fill_cylinder(c, rot, p, r):
    cx, cy = c
    a1 = [cx, cy, 0] #pts from non-rotated axis 
    a2 = [cx, cy, 1] #pts from non-rotated axis 
    a1 = rotate(rot, a1, o=c) # rotate axis pts to define cylinder edges
    a2 = rotate(rot, a2, o=c) # rotate axis pts to define cylinder edges
    in_cylinder = cyr_dist_fn(p=p, a1=a1, a2=a2) <= r
    return np.argwhere(in_cylinder)

def rotate(rot, p, o=None):
    if o is None:
        o = [0,0]
    p = np.copy(p)
    p[0] -= o[0]
    p[1] -= o[1]
    p = rot.apply(p)
    p[0] += o[0]
    p[1] += o[1]
    return p

def dist_cylinder(c, rot, p):
    cx, cy = c
    a1 = [cx, cy, 0] #pts from non-rotated axis 
    a2 = [cx, cy, 1] #pts from non-rotated axis 
    a1 = rotate(rot, a1, o=c) # rotate axis pts to define cylinder edges
    a2 = rotate(rot, a2, o=c) # rotate axis pts to define cylinder edges
    return cyr_dist_fn(p=p, a1=a1, a2=a2)

class GenerateVesselsAndSkinAndMasks:
    def __init__(
        self, resolutionXY=256, cylinder_size_max=30, cylinder_size_min=2, max_depth_cylinder_from_skin=90, lims_rot_x=80, lims_rot_y=30, skin_noise_min=10, skin_noise_max=40, behind_skin_noise=True, vessel_noise=True, prng=None, **kwargs
    ):
        self.resolutionXY = resolutionXY
        self.cylinder_size_max = cylinder_size_max
        self.cylinder_size_min = cylinder_size_min
        self.max_depth_cylinder_from_skin = max_depth_cylinder_from_skin
        self.lims_rot_x = lims_rot_x
        self.lims_rot_y = lims_rot_y
        self.skin_noise_min = skin_noise_min
        self.skin_noise_max = skin_noise_max
        self.behind_skin_noise = behind_skin_noise
        self.vessel_noise = vessel_noise
        self.prng = prng if prng is not None else np.random.RandomState(42)

        ### hidden params to fix routines which would normally be randomized for debugging purposes
        self.is_binary = kwargs.get('is_binary', False) ## overrides to make acoustic map binary
        self.num_cylinders = kwargs.get('num_cylinders', None) # overrides randomization of number of cylinders to a fixed number
        self.rot_x = kwargs.get('rot_x', None) # overrides randomization of rotation along x axis
        self.rot_y = kwargs.get('rot_y', None) # overrides randomization of rotation along y axis
        self.radius = kwargs.get('radius', None) # overrides rHorizontalMin and rHorizontalMax, which are derivatives of cylinder_size_min and cylinder_size_max
        self.depth_cylinder_from_skin = kwargs.get('depth_cylinder_from_skin', None) # overrides depth of the vessel center from the skinline to a fixed depth

    def generate(self):

        res_multiplier = self.resolutionXY / 256 ## deprecated
        rHorizontalMin = self.cylinder_size_min * res_multiplier
        rHorizontalMax = self.cylinder_size_max * res_multiplier
        max_depth_cylinder_from_skin = self.max_depth_cylinder_from_skin * res_multiplier
        maxDepthEllipseFromBorder = 30 * res_multiplier
        ## Parameters on cylinder rotation
        lims_rot_x = self.lims_rot_x #degrees [-lims_rot, lims_rot] rotation along x axis
        lims_rot_y = self.lims_rot_y #degrees , rotation along y axis

        # create mesh to use in line and ellipse
        x = np.linspace(0, self.resolutionXY - 1, self.resolutionXY)
        y = np.linspace(0, self.resolutionXY - 1, self.resolutionXY)
        meshX, meshY = np.meshgrid(x, y)
        pts = np.asarray(list(zip(meshX.flatten(), meshY.flatten(), np.zeros_like(meshX.flatten())))).T # points on x/y plane

        imageEllipses = np.zeros((self.resolutionXY, self.resolutionXY))
        imageEllipses_mask = np.zeros((self.resolutionXY, self.resolutionXY, 3))  # mask. channels: 0: skin line, 1: vessels

        ########################## create the skin line ############################################
        dist_skin_top = self.prng.randint(30,140)
        max_skin_bent = self.prng.randint(5,50)
        max_out_of_view_skin_ends = 100
        skin_left_end, skin_right_end = self.prng.randint(-1*max_out_of_view_skin_ends,0), int(self.resolutionXY)+self.prng.randint(0,max_out_of_view_skin_ends)
        p_coeff = np.polyfit(x=[skin_left_end, int(skin_left_end/2+skin_right_end/2), skin_right_end], y=[dist_skin_top+max_skin_bent, dist_skin_top, dist_skin_top+max_skin_bent], deg=2)
        p_fn = np.poly1d(p_coeff)
        x= np.arange(256, dtype=int)
        y = np.asarray(p_fn(x), dtype=int)
        lineCropped = np.zeros((256,256), dtype=float)
        lineCropped[x,y] = 1.
        skinlineGT = lineCropped
        if self.is_binary:
            pass
        else:
            lineCropped = gaussian_filter(lineCropped, sigma=1.5)
        lineCropped /= np.max(lineCropped) # rescale to [0,1]
        skinlineSmoothedGT = np.copy(lineCropped)
        #### create pattern behind skinline
        if self.behind_skin_noise:
            if self.is_binary:
                raise AssertionError('both behind_skin_noise and is_binary enabled. This makes no sense.')
            ## First add tail of a gauss
            n = NormDist(mu=0, sig=10, multiplier=0.5) 
            depth_skin_pattern = self.prng.randint(self.skin_noise_min, self.skin_noise_max)
            skin_pattern_dist_to_skin = self.prng.randint(-7,7)
            for i, d in enumerate(range(skin_pattern_dist_to_skin,skin_pattern_dist_to_skin+depth_skin_pattern)):
                lineCropped[x,y+i+3] = n(d)
            ## Add less structured speckle noise
            speckle_noise = self.prng.normal(loc=1, scale=0.1, size=lineCropped.shape)
            lineCropped *=speckle_noise 
            lineCropped /= np.max(lineCropped) #rescale to [0, 1]

        # add line to empty image first so it can check violation as well
        imageEllipses += lineCropped
        imageEllipses_mask[..., 0] += skinlineSmoothedGT

        skin_edge = [np.max(np.where(skinlineGT[r, :])) for r in range(self.resolutionXY)]  # for each row, find the min depth at which the ellipses can be drawn
        skin_region_edge = np.asarray(skin_edge)
        ######################################################################

        ## Initial prior on #cylinders
        if self.num_cylinders is None:
            if self.prng.choice([True, False]):
                num_cylinders = int(2)
            else:
                num_cylinders = int(
                    self.prng.poisson(lam=1.55) + 1
                )
        else:
            num_cylinders = self.num_cylinders

        image_pure_ellipses = np.copy(skinlineSmoothedGT) # includes smoothed skinline but excludes skin speckle noise

        for ind_cylinder in range(num_cylinders):

            is_violating = True
            while is_violating:  # Rough check to make sure no objects overlap.
                if self.rot_x is None:
                    rot_x = self.prng.uniform(-lims_rot_x, lims_rot_x)
                else:
                    rot_x = self.rot_x
                if self.rot_y is None:
                    rot_y = self.prng.uniform(-lims_rot_y, lims_rot_y)
                else:
                    rot_y = self.rot_y

                rot = scipy.spatial.transform.Rotation.from_euler('xy', [rot_x, rot_y], degrees=True) #rotation function of the z-axis aligned cylinder (vessel)
                if self.radius is None:
                    r = self.prng.uniform(rHorizontalMin, rHorizontalMax) #radius of the ellipse
                else: 
                    r = self.radius
                cy = self.prng.randint(self.resolutionXY) #cylinder y-center
                min_depth_vessel = np.argmax(skinlineGT[cy,:]) + r
                if self.depth_cylinder_from_skin is None:
                    cx = self.prng.randint(min_depth_vessel, min(min_depth_vessel + max_depth_cylinder_from_skin, self.resolutionXY - maxDepthEllipseFromBorder)) #cylinder x-center
                else: 
                    cx = int(self.depth_cylinder_from_skin)
                c = (cx, cy)

                ######################################################
                if self.is_binary:
                    dist_ = dist_cylinder(c, rot, p=pts) <= r ## value at 1. cylinder center = 1, cylinder contour = 1, outside cylinder = 0
                    dist_ = np.asarray(dist_, dtype=float) 
                    singleImage = dist_.reshape(meshX.shape)
                else:
                    vessel_base_type_rand = self.prng.rand()
                    if vessel_base_type_rand < 0.5:
                        ## Option 1: fill all vessel with constant value #### HOMOGENEOUS VESSELS ####
                        # pts_in = fill_cylinder(c, rot, p=pts, r=r) # check points from image plane (x-y plane at z=0) that fall within the rotated cylinder
                        # ptx, pty = np.unravel_index(pts_in, shape=meshX.shape) # get back (x,y) indices
                        # singleImage = np.zeros_like(imageEllipses)
                        # singleImage[ptx, pty] = 1.
                        dist_ = dist_cylinder(c, rot, p=pts) <= r ## value at 1. cylinder center = 1, cylinder contour = 1, outside cylinder = 0
                        dist_ = np.asarray(dist_, dtype=float) 
                        singleImage = dist_.reshape(meshX.shape)
                    # elif vessel_base_type_rand >= 0.5:
                    else:
                        ## Option 2: fill vessel with linearly descreasing value from center ot contours #### EXPONENTIALLY/LINEARLY/LOGARITHMICALLY ATTENUATING VESSELS FROM CENTER ####
                        dist_ = r - dist_cylinder(c, rot, p=pts) ## value at 1. cylinder center = r, cylinder contour = 0, outside cylinder <0
                        dist_[dist_ < 0] = 0
                        # dist_ = np.log(dist_+1e-3) # Make the attenuation logarithmic
                        dist_ = dist_**0.3 # Make the attenuation logarithmic
                        # dist_ = dist_**1.5 # Make the attenuation exponential (makes most large vessels almost disappear)
                        singleImage = dist_.reshape(meshX.shape)
                        singleImage -= np.min(singleImage)
                        singleImage /= np.max(singleImage) # rescale vessel to [0,1]
                        #######################################################

                if self.is_binary:
                    pass
                else:
                    ## apply gauss filter to smoothen edges (to look similar to previous vessels)
                    # Draw a wide gauss near higher sigma (50%) or a sharp gauss near low sigma (50%)
                    if self.prng.rand() > 0.5:
                        s_ = np.max([self.prng.normal(1., 1), 0.01])
                    else:
                        s_ = np.max([self.prng.normal(0.2, 0.2), 0.01])
                    singleImage = gaussian_filter(singleImage, sigma=s_) ## causing some issues with the post - binarization for some applications
                    singleImage /= np.max(singleImage)
                    singleImage[singleImage<0.1] = 0. ## measure to try and remove gaussian filter artifacts far from vessels.  
                    
                    apply_noise_on_vessel_rand = self.prng.rand()
                    if self.vessel_noise:
                        if apply_noise_on_vessel_rand < 0.5:
                            ## apply normal noise inside vessels
                            speckle_noise = self.prng.normal(loc=1, scale=0.0133, size=singleImage.shape)
                            singleImage *= speckle_noise 
                        else:
                            ## do not apply normal noise inside vessels
                            pass
                    singleImage /= np.max(singleImage)/0.9

                is_violating = (
                    np.sum(
                        np.logical_and(
                            image_pure_ellipses.flatten() != 0.0, singleImage.flatten() != 0.0
                        )
                    )
                    > 0.0
                )

            for r in range(len(skin_region_edge)): ## Loop determines the lowest point of skin+vessel for each row. Needed for skin_region mask
                p_ = np.argwhere(singleImage[r] > 0).flatten()
                if len(p_) > 0:
                    skin_region_edge[r] = np.max((skin_region_edge[r], np.max(p_)))

            imageEllipses += singleImage #sum up
            # imageEllipses[np.where(singleImage>0)] = singleImage[np.where(singleImage>0)] #ignore existing background when adding vessel (only matters for the under skin normal noise)
            image_pure_ellipses += singleImage
            imageEllipses_mask[..., 1] += singleImage

        # imageEllipses = imageEllipses[:, ::-1, ...]  # put the skin on the right side to get the same artifact effects as in vivo images.
        # imageEllipses_mask = imageEllipses_mask[:, ::-1, ...]
        imageClipped = np.clip(imageEllipses, 0, 1)
        imageEllipses_mask = np.clip(imageEllipses_mask, 0, 1)
        imageClipped = np.transpose(imageClipped, (1, 0))
        imageEllipses_mask = np.transpose(imageEllipses_mask, (1, 0, 2))
        return imageClipped, imageEllipses_mask

def process_vessel_and_skinline(v, sl):
    slb = np.asarray(sl>0.5, dtype=bool)
    vb = np.asarray(v>0., dtype=bool)
    im = np.zeros_like(vb, dtype=np.uint8)
    im[vb] = 1
    im[slb] = 2
    return im