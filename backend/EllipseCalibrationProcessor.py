import py4DSTEM
import numpy as np
import cv2
class EllipseCalibrationProcessor:
    """椭圆校准处理器类"""
    
    __slots__ = [
        '_datacube',
        '_bragg_peaks', 
        '_bragg_vector_map',
        '_p_ellipse',
        '_q_range',
        '_dp_mean',
        '_dp_max',
        '_probe_semiangle',
        '_probe_qx0',
        '_probe_qy0',
        '_center',
        '_plot_mean_diffraction_pattern_params',
        '_r_inner',
        '_r_outer',
        '_radii',
        '_plot_ADF_detector_params',
        '_plot_ADF_image_params',
        '_cal_ADF_image_params',
        '_ADF_image',
        '_probe',
        '_get_synthetic_probe_params',
        '_get_kernel_params',
        '_kernel',
        '_plot_kernel_params',
        '_rxs',
        '_rys',
        '_colors',
        '_detect_params',
        '_selected_disks',
        '_plot_image_grid_params',
        '_bragg_peaks',
        '_qxy_origins',
        '_qx0_fit',
        '_qy0_fit',
        '_qx0_residuals',
        '_qy0_residuals',
        '_bragg_vector_map_centered',
        '_q_range',
        '_p_ellipse',
        '_bragg_vector_map_centered_corrected'
    
    ]
    
    def load_data(self, filepath):
        """加载4D-STEM数据并初始化参数
        
        Args:
            filepath (str): 数据文件路径
        """
        self._datacube = py4DSTEM.import_file(filepath=filepath)
        self._bragg_peaks = None
        self._bragg_vector_map = None 
        self._p_ellipse = None
        self._q_range = (41, 48) # 默认的q范围
        self._dp_mean = None
        self._dp_max = None
        self._probe_semiangle = None
        self._probe_qx0 = None
        self._probe_qy0 = None
        self._center = None
        self._plot_mean_diffraction_pattern_params = {
            'figsize': (4, 4),
            'circle': {
                'center': self._center,
                'R': self._probe_semiangle
            },
            'ticks': False,
            'returnfig': True,
            'vmax': 1,
        }
        self._r_inner = None
        self._r_outer = None
        self._radii = None
        self._plot_ADF_detector_params = {
            'mode': 'annular',
            'geometry': (self._center, self._radii),
            'figsize': (4, 4),
            'ticks': False,
        }
        self._plot_ADF_image_params = {
            'figsize': (4, 4),
        }
        self._cal_ADF_image_params = {
            'mode': 'annular',
            'geometry': (self._center, self._radii),
            'name': 'dark_field',
        }
        self._ADF_image = None
        self._probe = None
        self._get_synthetic_probe_params = {
            'radius': self._probe_semiangle,
            'width': 0.7,
            'Qshape': self._datacube.Qshape,
        }
        self._get_kernel_params = {
            'mode': 'sigmoid',
            'origin': (self._datacube.Qshape[0]/2, self._datacube.Qshape[0]/2),
            'radii': (self._probe_semiangle*1.2, self._probe_semiangle*4)
        }
        self._kernel = None
        self._plot_kernel_params = {
            'R': 30,
            'L': 30,
            'W': 1
        }
        self._rxs = 35, 35, 62, 84, 75, 62
        self._rys = 50, 62, 70, 40, 100, 120
        self._colors = ['r','limegreen','c','g','orange', 'violet']
        self._detect_params = {
            'minAbsoluteIntensity': 0,   # intensity threshold
            'minRelativeIntensity': 0,   # int. thresh. relative to brightest disk in each pattern
            'minPeakSpacing': 10,         # if two peaks are closer than this (in pixels), remove the dimmer peak
            'edgeBoundary': 60,           # remove peaks within this distance of the edge of the diffraction pattern
            'sigma': 0,                  # gaussian blur size to apply to cross correlation before finding maxima
            'maxNumPeaks': 100,          # maximum number of peaks to return, in order of intensity
            'subpixel' : 'poly',         # subpixel resolution method
            'corrPower': 1.0,            # if <1.0, performs a hybrid cross/phase correlation. More sensitive to edges and to noise
            'CUDA': False,              # if a GPU is configured and cuda dependencies are installed, speeds up calculation 
        }
        self._selected_disks = None    
        self._plot_image_grid_params = {
            'get_ar': lambda i:self._datacube[self._rxs[i],self._rys[i],:,:],
            'H': 2, 
            'W': 3,
            'axsize': (5,5),
            'scaling': 'power',
            'power': 0.75,
            'vmin': 0.1, 
            'vmax': 0.999,
            'get_bordercolor': lambda i:self._colors[i],
            'get_x': lambda i: self._selected_disks[i].data['qx'],
            'get_y': lambda i: self._selected_disks[i].data['qy'],
            'get_pointcolors': lambda i: self._colors[i],
            'open_circles': True,
            'scale': 100,
            # hist=True
        }
        self._bragg_peaks = None
        self._qxy_origins = None
        self._qx0_fit = None
        self._qy0_fit = None
        self._qx0_residuals = None
        self._qy0_residuals = None
        self._bragg_vector_map_centered = None
        self._q_range = (41, 48)
        self._p_ellipse = None
        self._bragg_vector_map_centered_corrected = None
    def get_ellipse_calibration(self):
        """获取椭圆校准"""
        self._bragg_peaks.calibration.set_p_ellipse(self._p_ellipse)
        self._bragg_peaks.setcal()
        self._bragg_vector_map_centered_corrected = self._bragg_peaks.get_bvm()
        fig = py4DSTEM.show(
            self._bragg_vector_map_centered_corrected,
            figsize = (4,4),
        )
        return fig[0]
            
    def get_elliptical_fit(self):
        """获取椭圆拟合"""
        fig = py4DSTEM.visualize.show_elliptical_fit(
            self._bragg_vector_map_centered,
            self._q_range,
            self._p_ellipse,
            cmap = 'gray',
        )
        return fig[0]

    def set_p_ellipse(self):
        """设置椭圆参数"""
        self._p_ellipse = py4DSTEM.process.calibration.fit_ellipse_1D(
            self._bragg_vector_map_centered,
            center = self._bragg_vector_map_centered.origin,
            fitradii = self._q_range,
        )

    def get_specific_bragg_peaks(self):
        """获取特定布拉格峰"""
        fig = py4DSTEM.show(
            self._bragg_vector_map_centered,
            cmap = 'gray',
            annular = {
                'center': self._bragg_vector_map_centered.origin,
                'radii': self._q_range, 'fill':True, 'color':'r', 'alpha':0.3
            }
        )
        return fig[0]

    def get_bragg_vector_map_centered(self):
        """获取布拉格向量图"""
        self._bragg_vector_map_centered = self._bragg_peaks.get_bvm()
        fig = py4DSTEM.show(
            self._bragg_vector_map_centered,
            figsize = (4,4),
        )
        return fig[0]

    def set_qx0_qy0_fit_residuals(self):
        """设置qx0和qy0的拟合和残差"""
        self._qx0_fit, self._qy0_fit, self._qx0_residuals, self._qy0_residuals = self._bragg_peaks.fit_ellipse(
            figsize = (4,4)
        )

    def set_qxy_origins(self):
        """设置qxy原点"""
        self._qxy_origins = self._bragg_peaks.measure_origin(
            score_method = 'intensity weighted distance',
            center_guess = [118, 128]
        )
        
    def get_bragg_peaks(self):
        """获取布拉格峰"""
        self._bragg_peaks = self._datacube.find_Bragg_disks(
            template = self._kernel,
            **self._detect_params,
        )

    def get_image_grid(self):
        """获取图像网格"""
        fig = py4DSTEM.visualize.show_image_grid(
            **self._plot_image_grid_params
        )
        return fig[0]
        
    def get_selected_disks(self):
        """获取选中的磁盘"""
        self._selected_disks = self._datacube.find_Bragg_disks(
            data = (self._rxs, self._rys),
            template = self._kernel,
            **self._detect_params,
        )
    
    def get_kernel_plot(self):
        """获取核函数图像"""
        fig = py4DSTEM.visualize.show(
            self._kernel,
            **self._plot_kernel_params
        )
        return fig[0]
    
    def get_synthetic_probe(self):
        """获取合成探针"""
        self._probe = py4DSTEM.Probe.generate_synthetic_probe(
            **self._get_synthetic_probe_params
        )
    
    def get_kernel(self):
        """获取核函数"""
        self._kernel = self._probe.get_kernel(
            **self._get_kernel_params
        )
    
    def get_normalized_ADF_image(self):
        """获取归一化后的ADF图像"""
        img = np.zeros([256, 256])
        for i in range(256):
            for j in range(256):
                img[i, j] = self._ADF_image[i, j]

        img = np.floor((img - np.min(img)) / (np.max(img) - np.min(img)) * 256)
        img[img >= 256] = 255
        img = img.astype(np.uint8)
        
        cv2.imwrite("./test.png", img)

    def get_ADF_image(self):
        """获取ADF图像"""
        fig = py4DSTEM.show(
            self._ADF_image,
            **self._plot_ADF_image_params
        )
        return fig[0]

    def cal_ADF_image(self):
        """计算ADF图像"""
        self._ADF_image = self._datacube.get_virtual_image(
            **self._cal_ADF_image_params
        )
        
    def get_ADF_detector(self):
        """获取ADF探测器"""
        fig = self._datacube.position_detector(
            **self._plot_ADF_detector_params
        )
        return fig[0]
    
    def set_radii(self):
        """设置椭圆半径"""
        self._r_inner = self._probe_semiangle * 3
        self._r_outer = self._probe_semiangle * 6
        self._radii = (self._r_inner, self._r_outer)
        
    def set_probe_data(self):
        """设置探针数据"""
        self._dp_mean = self._datacube.get_dp_mean()
        self._probe_semiangle, self._probe_qx0, self._probe_qy0 = self._datacube.get_probe_size(self._dp_mean)
        
    def set_center(self):
        """设置中心"""
        self.center = (self._probe_qx0, self._probe_qy0)
        
    def get_dp_mean(self):
        """获取平均衍射图"""
        self._dp_mean = self._datacube.get_dp_mean()
        fig = py4DSTEM.visualize.show(
            self._dp_mean,
            figsize=(4,4),
            ticks=False,
        )
        return fig[0]
    
    def get_dp_max(self):
        """获取最大衍射图"""
        self._dp_max = self._datacube.get_dp_max()
        fig = py4DSTEM.visualize.show(
            self._dp_max,
            figsize=(4,4),
            ticks=False,
        )
        return fig[0]
    
    def get_mean_diffraction_pattern(self):
        """获取平均衍射图"""
        fig, ax = py4DSTEM.visualize.show(
            self._dp_mean,
            **self._plot_mean_diffraction_pattern_params
        )
        return fig[0]
    
    def get_estimate_probe_radius(self):
        """获取估计的探针半径"""
        print('Estimated probe radius =', '%.2f' % self._probe_semiangle, 'pixels')
