import numpy as np

from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from senfuslib import NamedArray, AtIndex
from solution import quaternion as quaternion_solu


@dataclass
class RotationQuaterion(NamedArray):
    """Class representing a rotation quaternion (norm = 1). Has some useful
    methods for converting between rotation representations.

    Args:
        real_part (float): eta (n) in the book, w in scipy notation
        vec_part (ndarray[3]): epsilon in the book, (x,y,z) in scipy notation
    """
    eta: AtIndex[0]
    epsilon: AtIndex[1:4]

    def __post_init__(self):
        norm = np.sqrt(self.eta**2 + sum(self.epsilon**2))
        if not np.allclose(norm, 1):
            self.eta /= norm
            self.epsilon /= norm
        if self.eta < 0:
            self.eta *= -1
            self.epsilon *= -1

    def multiply(self, other: 'RotationQuaterion') -> 'RotationQuaterion':
        """Multiply two rotation quaternions
        Hint: see (10.33)

        As __matmul__ is implemented for this class, you can later use:
        q1@q2 which is equivalent to q1.multiply(q2)

        Args:
            other: the other quaternion    
        Returns:
            quaternion_product: the product
        """
        eta_a, epsilon_a = self
        eta_b, epsilon_b = other

        def skew(vector):
            skewMatrix = np.array([
                [0, -vector[2], vector[1]],
                [vector[2], 0, -vector[0]],
                [-vector[1], vector[0], 0]
            ])
            return skewMatrix

        m1 = np.vstack((
            np.hstack((eta_a, -epsilon_a.T)),
            np.hstack((epsilon_a.reshape((3, 1)), eta_a*np.eye(3) + skew(epsilon_a)))
        ))
        m2 = np.hstack((eta_b, epsilon_b))
        q = m1@m2
        eta_out = q[0]
        epout = q[1:]

        return RotationQuaterion(eta_out, epout)

    def conjugate(self) -> 'RotationQuaterion':
        """Get the conjugate of the RotationQuaternion"""
        return RotationQuaterion(self.eta, -self.epsilon)

    def diff(self, other: 'RotationQuaterion') -> 'RotationQuaterion':
        """Get the difference between two quaternions3
        So that self @ self.diff(other) == other"""
        return self.conjugate()@other

    def diff_as_avec(self, other: 'RotationQuaterion') -> 'np.ndarray[3]':
        """Get the difference between two quaternions as a rotation vector"""
        return self.diff(other).as_avec()

    def as_rotmat(self) -> 'np.ndarray[3, 3]':
        """Get the rotation matrix representation of self

        Returns:
            R (ndarray[3,3]): rotation matrix
        """
        R = Rotation.from_quat(self._as_scipy_quat()).as_matrix()
        return R

    @property
    def R(self) -> 'np.ndarray[3, 3]':
        return self.as_rotmat()

    def as_euler(self) -> 'np.ndarray[3]':
        """Get the euler angle representation of self

        Returns:
            euler (ndarray[3]): extrinsic xyz euler angles (roll, pitch, yaw)
        """
        euler = Rotation.from_quat(self._as_scipy_quat()).as_euler('xyz')
        return euler

    def as_avec(self) -> 'np.ndarray[3]':
        """Get the angles vector representation of self

        Hint: this is most often called rotation vector or rotvec. 

        Returns:
            avec (ndarray[3]):  3 dimensional vector which is co-directional to
                the axis of rotation and whose norm gives the angle of rotation
        """
        avec = Rotation.from_quat(self._as_scipy_quat()).as_rotvec()
        return avec

    @staticmethod
    def from_avec(avec: 'np.ndarray[3]') -> 'RotationQuaterion':
        """Create a RotationQuaternion from an angle vector

        Args:
            avec (ndarray[3]): 3 dimensional vector which is co-directional to
                the axis of rotation and whose norm gives the angle of rotation
        """
        scipy_quat = Rotation.from_rotvec(avec).as_quat()
        return RotationQuaterion(scipy_quat[3], scipy_quat[:3])

    @staticmethod
    def from_euler(euler: 'np.ndarray[3]') -> 'RotationQuaterion':
        """Get a rotation quaternion from euler angles
        usage: rquat = RotationQuaterion.from_euler(euler)

        Args:
            euler (ndarray[3]): extrinsic xyz euler angles (roll, pitch, yaw)

        Returns:
            rquat (RotationQuaternion): the rotation quaternion
        """
        scipy_quat = Rotation.from_euler('xyz', euler).as_quat()
        return RotationQuaterion(scipy_quat[3], scipy_quat[:3])

    def _as_scipy_quat(self) -> 'np.ndarray[4]':
        """If you're using scipys Rotation class, this can be handy"""
        return np.append(self.epsilon, self.eta)

    def __iter__(self):
        return iter([self.eta, self.epsilon])

    def __matmul__(self, other) -> 'RotationQuaterion':
        """Lets u use the @ operator, q1@q2 == q1.multiply(q2)"""
        return self.multiply(other)
