import py4DSTEM
import numpy as np


class OrientationProcessor:
    __slots__ = [
        "dataset",  # 存储加载的数据集
        "_orientation_plan",  # 取向计划对象
        "_orientation_map",  # 计算生成的取向映射
        "_strain_map",  # 应变映射对象
        "_grain_boundaries",  # 晶界数据
        "_rxs",  # 选择区域的x坐标列表
        "_rys",  # 选择区域的y坐标列表
        "_colors",  # 颜色映射，用于可视化
        "_orientation_params",  # 用于创建取向计划的参数
        "_strain_params",  # 用于计算应变的参数
        "_grain_boundary_params",  # 用于提取晶界的参数
        "_ellipse_fit_range",  # 用于拟合椭圆的范围
        "_bragg_peaks",  # Bragg峰数据
        "_bvm_raw",  # 原始Bragg矢量图
        "_bvm_centered",  # 对齐中心后的Bragg矢量图
        "_amorph_ROI",  # 无序环区域
        "_p_ellipse",  # 椭圆拟合结果参数
        "_p_dsg",  # 椭圆拟合图形化参数
        "_adf",  # 暗场图像
        "_bf",  # 亮场图像
        "_probe",  # 探针数据
        "_kernel",  # 探针核
        "_probe_params",  # 探针的初始化参数
        "_disk_detect_params",  # Bragg盘检测的参数
    ]
    def __init__(self):
        self.dataset = None
        self.orientation_plan = None
        self.orientation_map = None
        self.strain_map = None
        self.grain_boundaries = None

    def load_dataset(self, file_path):
        """Load the dataset from a file."""
        self.dataset = py4DSTEM.import_file(file_path)

    def create_orientation_plan(self, params):
        """
        Create an orientation plan for crystal alignment.

        Args:
            params (dict): Parameters for orientation plan generation.
        """
        self.orientation_plan = self.dataset.create_orientation_plan(**params)

    def execute_orientation_plan(self):
        """
        Execute the orientation plan to compute orientations.
        """
        if self.orientation_plan is None:
            raise ValueError("Orientation plan has not been created.")
        self.orientation_map = self.orientation_plan.execute()

    def visualize_orientation_map(self, orientation_ind=0, show_axes=False):
        """
        Visualize the computed orientation map.

        Args:
            orientation_ind (int): Index of the orientation to visualize.
            show_axes (bool): Whether to show axes in the visualization.
        """
        if self.orientation_map is None:
            raise ValueError("Orientation map has not been computed.")
        fig = py4DSTEM.visualize.plot_orientation_map(
            self.orientation_map, orientation_ind=orientation_ind, show_axes=show_axes, returnfig=True
        )
        return fig

    def compute_strain_map(self, strain_params):
        """
        Compute the strain map based on orientation data.

        Args:
            strain_params (dict): Parameters for strain calculation.
        """
        if self.orientation_map is None:
            raise ValueError("Orientation map has not been computed.")
        self.strain_map = py4DSTEM.StrainMap(braggvectors=self.orientation_map, **strain_params)

    def visualize_strain_map(self):
        """
        Visualize the computed strain map.
        """
        if self.strain_map is None:
            raise ValueError("Strain map has not been computed.")
        fig = py4DSTEM.visualize.plot_strain_map(self.strain_map, returnfig=True)
        return fig

    def extract_grain_boundary_data(self, params):
        """
        Extract grain boundary data based on the orientation map.

        Args:
            params (dict): Parameters for grain boundary extraction.
        """
        if self.orientation_map is None:
            raise ValueError("Orientation map has not been computed.")
        self.grain_boundaries = self.dataset.extract_grain_boundaries(self.orientation_map, **params)
        return self.grain_boundaries

    def visualize_grain_boundaries(self):
        """
        Visualize the extracted grain boundaries.
        """
        if self.grain_boundaries is None:
            raise ValueError("Grain boundaries have not been extracted.")
        fig = py4DSTEM.visualize.plot_grain_boundaries(self.grain_boundaries, returnfig=True)
        return fig

    def save_results(self, save_path):
        """
        Save the computed results to a file.

        Args:
            save_path (str): Path to save the results.
        """
        if self.orientation_map is None:
            raise ValueError("No orientation map to save.")
        py4DSTEM.save(save_path, self.orientation_map, mode="o")

    def create_virtual_image(self, mode, geometry, name):
        """
        Create a virtual image (bright or dark field).

        Args:
            mode (str): "circle" for BF, "annulus" for ADF.
            geometry (tuple): Geometry definition for virtual image.
            name (str): Name of the virtual image.
        """
        if self.dataset is None:
            raise ValueError("Dataset has not been loaded.")
        return self.dataset.get_virtual_image(mode=mode, geometry=geometry, name=name)

    def detect_bragg_disks(self, template, detect_params):
        """
        Detect Bragg disks in the diffraction data.

        Args:
            template: Template for disk detection.
            detect_params (dict): Parameters for disk detection.
        """
        if self.dataset is None:
            raise ValueError("Dataset has not been loaded.")
        self.bragg_disks = self.dataset.find_Bragg_disks(template=template, **detect_params)

    def visualize_bragg_disks(self, rxs, rys, colors, scale=300):
        """
        Visualize detected Bragg disks on selected diffraction patterns.

        Args:
            rxs (list): List of x-indices for visualization.
            rys (list): List of y-indices for visualization.
            colors (list): Colors for marking the disks.
            scale (int): Scale for visualization.
        """
        fig = py4DSTEM.visualize.show_image_grid(
            get_ar=lambda i: self.dataset.data[rxs[i], rys[i], :, :],
            H=2,
            W=3,
            get_bordercolor=lambda i: colors[i],
            get_x=lambda i: self.bragg_disks[i].data["qx"],
            get_y=lambda i: self.bragg_disks[i].data["qy"],
            get_pointcolors=lambda i: colors[i],
            open_circles=True,
            scale=scale,
            returnfig=True,
        )
        return fig

    def fit_ellipse_amorphous_ring(self, amorphous_diffraction, fit_radii):
        """
        Fit an ellipse to an amorphous diffraction ring.

        Args:
            amorphous_diffraction: Amorphous diffraction pattern.
            fit_radii (tuple): Radii range for ellipse fitting.
        """
        center = amorphous_diffraction.calibration.get_origin_mean()
        self.p_ellipse, self.p_dsg = py4DSTEM.process.calibration.fit_ellipse_amorphous_ring(
            data=amorphous_diffraction.data, center=center, fitradii=fit_radii
        )

    def visualize_ellipse_fit(self, amorphous_diffraction, fit_radii):
        """
        Visualize the ellipse fit on the amorphous diffraction ring.

        Args:
            amorphous_diffraction: Amorphous diffraction pattern.
            fit_radii (tuple): Radii range for ellipse fitting.
        """
        fig = py4DSTEM.visualize.show_amorphous_ring_fit(
            amorphous_diffraction.data,
            fitradii=fit_radii,
            p_dsg=self.p_dsg,
            returnfig=True,
        )
        return fig

