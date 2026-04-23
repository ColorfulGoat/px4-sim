import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from sklearn.linear_model import RANSACRegressor
import struct

class PointCloudNode(Node):
    def __init__(self):
        super().__init__('point_cloud_node')

        self.subscription = self.create_subscription(
            PointCloud2,
            '/depth_camera/points',
            self.pc_callback,
            10)

        self.pub_filtered = self.create_publisher(
            PointCloud2, '/pointcloud/obstacles', 10)

        self.pub_ground = self.create_publisher(
            PointCloud2, '/pointcloud/ground', 10)

        self.get_logger().info('Point Cloud Node started!')

    def pc_callback(self, msg):
        # Convert PointCloud2 to numpy array
        points = []
        for p in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            points.append([p[0], p[1], p[2]])

        if len(points) < 10:
            self.get_logger().warn('Not enough points')
            return

        pts = np.array(points, dtype=np.float32)

        # Remove inf and nan values
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]

        # Remove points too far away (depth camera noise)
        mask2 = (np.abs(pts[:, 0]) < 50) & (np.abs(pts[:, 1]) < 50) & (np.abs(pts[:, 2]) < 50)
        pts = pts[mask2]

        if len(pts) < 10:
            self.get_logger().warn('Not enough valid points after filtering')
            return
        
        self.get_logger().info(f'Total points: {len(pts)}')

        # RANSAC ground plane segmentation
        # Fit plane: z = ax + by + c
        try:
            X = pts[:, :2]  # x, y
            z = pts[:, 2]   # z

            ransac = RANSACRegressor(residual_threshold=0.1, max_trials=100)
            ransac.fit(X, z)

            inlier_mask = ransac.inlier_mask_
            outlier_mask = ~inlier_mask

            ground_pts = pts[inlier_mask]
            obstacle_pts = pts[outlier_mask]

            self.get_logger().info(
                f'Ground: {len(ground_pts)} pts | Obstacles: {len(obstacle_pts)} pts')

            # Publish obstacle cloud
            if len(obstacle_pts) > 0:
                obs_msg = self.numpy_to_pc2(obstacle_pts, msg.header)
                self.pub_filtered.publish(obs_msg)

            # Publish ground cloud
            if len(ground_pts) > 0:
                gnd_msg = self.numpy_to_pc2(ground_pts, msg.header)
                self.pub_ground.publish(gnd_msg)

        except Exception as e:
            self.get_logger().error(f'RANSAC error: {str(e)}')

    def numpy_to_pc2(self, pts, header):
        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        cloud_data = []
        for p in pts:
            cloud_data.append(struct.pack('fff', float(p[0]), float(p[1]), float(p[2])))

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(pts)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * len(pts)
        msg.data = b''.join(cloud_data)
        msg.is_dense = True
        return msg

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
