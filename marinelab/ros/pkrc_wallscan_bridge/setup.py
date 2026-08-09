from setuptools import setup

package_name = "pkrc_wallscan_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mjkim",
    maintainer_email="xiahholic5@gmail.com",
    description="Sensor topics to WallFrameStateEstimator bridge for the PKRC wallscan",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "estimator_bridge = pkrc_wallscan_bridge.estimator_bridge:main",
            "wallscan_controller = pkrc_wallscan_bridge.wallscan_controller:main",
            "thrust_mapper = pkrc_wallscan_bridge.thrust_mapper:main",
        ],
    },
)
