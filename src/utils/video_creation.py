r"""
Video creation from the snapshot files written during a simulation.

Each snapshot file holds one frame of column-density (``R-*-cd.dat``) or
velocity (``V-*-cd.dat``) data. The functions here read those frames back,
map them to an 8-bit colour image and write an MP4 at 10 frames per second.

Cartesian data is already a rectangular :math:`(n_1, n_3)` image. Cylindrical
data is stored on the polar grid :math:`(n_r, n_\varphi)` and is resampled onto
a square Cartesian image first, so the condensate appears as a disk rather than
as an unrolled strip.
"""
##################################################################################
###################### CREATE VIDEO ##############################################
##################################################################################
import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc


def _polar_to_cartesian_map(n_r, n_phi, img_size):
    r"""
    Precompute the nearest-neighbour lookup from a Cartesian pixel grid to
    polar :math:`(r, \varphi)` indices.

    The circular domain :math:`r \in [0, r_\mathrm{max}]` is mapped onto a
    square image of side ``img_size``. Writing :math:`(x, y)` for the pixel
    coordinates rescaled to :math:`[-1, 1]`, the lookup is

    .. math::

        i_r = \Bigl\lfloor n_r \sqrt{x^{2} + y^{2}} \Bigr\rfloor,
        \qquad
        i_\varphi = \Bigl\lfloor \frac{n_\varphi}{2\pi}
            \bigl[\operatorname{atan2}(y, x) \bmod 2\pi\bigr] \Bigr\rfloor,

    with both indices clipped to their valid range. Pixels beyond the disk
    boundary, :math:`\sqrt{x^{2} + y^{2}} > 1`, are flagged instead of being
    clipped into the data.

    Args:
        n_r (int): Number of radial grid points in the snapshot files.
        n_phi (int): Number of azimuthal grid points in the snapshot files.
        img_size (int): Side length of the square output image, in pixels.

    Returns:
        tuple: ``(i_r, i_phi, outside)`` — the two integer index arrays of
        shape ``(img_size, img_size)``, and a boolean mask that is ``True`` for
        pixels beyond the disk boundary.
    """
    y_idx, x_idx = np.mgrid[0:img_size, 0:img_size]
    x = (x_idx - img_size / 2.0) / (img_size / 2.0)   # [-1, 1]
    y = (y_idx - img_size / 2.0) / (img_size / 2.0)   # [-1, 1]
    r_norm = np.sqrt(x ** 2 + y ** 2)
    phi_map = np.arctan2(y, x) % (2.0 * np.pi)
    i_r   = np.clip((r_norm * n_r).astype(int),   0, n_r   - 1)
    i_phi = np.clip((phi_map / (2.0 * np.pi) * n_phi).astype(int), 0, n_phi - 1)
    return i_r, i_phi, r_norm > 1.0


def _colorise(data_2d):
    r"""
    Apply the jet-like colour mapping used by :func:`create_video` and return a
    BGR image.

    The data is first shifted and rescaled to the full 8-bit range,

    .. math::

        I = \Bigl\lfloor \frac{255\,(d - \min d)}{\max (d - \min d)}
            \Bigr\rfloor \in [0, 255],

    and a flat frame (:math:`\max d = \min d`) maps to all zeros. The three
    channels are then piecewise-linear functions of :math:`I`:

    .. math::

        R = \begin{cases} I, & I < 230 \\ 255, & \text{otherwise} \end{cases}
        \qquad
        G = \begin{cases} I, & I \le 127 \\ 255 - I, & \text{otherwise}
            \end{cases}
        \qquad
        B = \begin{cases} 255 - I, & I > 40 \\ 0, & \text{otherwise}
            \end{cases}

    Args:
        data_2d (numpy.ndarray): 2-D array of frame data.

    Returns:
        numpy.ndarray: BGR image of dtype ``uint8``, shape
        ``data_2d.shape + (3,)``, in the channel order OpenCV expects.
    """
    zeroed = data_2d - np.min(data_2d)
    max_val = np.max(zeroed)
    if max_val == 0:
        intImg = np.zeros(data_2d.shape, dtype=np.uint8)
    else:
        intImg = np.uint8((255.0 / max_val) * zeroed)
    intImgR = np.where(intImg < 230,   intImg,       255)
    intImgG = np.where(intImg <= 127,  intImg,       255 - intImg)
    intImgB = np.where(intImg > 40,    255 - intImg, 0)
    return np.dstack((intImgB, intImgG, intImgR))

def create_video(count,\
                 simulation_name,\
                 n1,n3
                 ):
  r"""
  Create a video from the Cartesian column-density snapshots.

  The third column of each ``R-*-cd.dat`` file holds the column density
  :math:`n(x, z)` on the :math:`(n_1, n_3)` grid; every frame is colour-mapped
  and written to ``{simulation_name}_fps10_frame{count}.mp4``.

  Args:
      count (int): Total number of frames, i.e. of snapshot files to read.
      simulation_name (str): Base name for the output file.
      n1 (int): Number of grid points along x.
      n3 (int): Number of grid points along z.
  """

  FPS=10
  SimulationName = simulation_name + f'_fps{FPS}_frame{count}'
  VideoDims=(n1,n3)
  frames=count
  video=VideoWriter(f'{SimulationName}.mp4', 0x7634706d, float(FPS), VideoDims)

  for framenum in range(frames):
    file_path = f'R-{framenum:003}-cd.dat'
    file = open(file_path,'r')
    img=np.reshape(np.loadtxt(file, delimiter=',', usecols=2),VideoDims)
    file.close()

    zeroed=img-np.min(img)*np.ones(img.shape)
    intImg= np.uint8((255/np.max(zeroed))*zeroed)

    intImgR = np.where(intImg<230 , intImg, 255 )
    intImgG = np.where((intImg<=127) ,intImg, 255-intImg)
    intImgB = np.where((intImg>40) , 255-intImg, 0)
    img3=np.dstack((intImgB,intImgG,intImgR))

    video.write(img3)
  video.release()


def create_video_cylindrical(count, simulation_name, n_r, n_phi, img_size=None):
  r"""
  Create a video from the cylindrical column-density snapshots.

  The :math:`(n_r, n_\varphi)` polar data of the ``R-*-cd.dat`` files is
  resampled onto a square Cartesian image with
  :func:`_polar_to_cartesian_map`, so the condensate appears as a disk rather
  than as an unrolled strip. Pixels outside the disk are set to zero.

  Args:
      count (int): Total number of frames.
      simulation_name (str): Base name for the output file.
      n_r (int): Number of radial grid points, matching the snapshot files.
      n_phi (int): Number of azimuthal grid points, matching the snapshot
          files.
      img_size (int): Side length of the square output image in pixels.
          Defaults to ``2 * n_r``.
  """
  if img_size is None:
      img_size = 2 * n_r

  FPS = 10
  SimulationName = simulation_name + f'_fps{FPS}_frame{count}'
  video = VideoWriter(f'{SimulationName}.mp4', 0x7634706d, float(FPS), (img_size, img_size))

  i_r, i_phi, outside = _polar_to_cartesian_map(n_r, n_phi, img_size)

  for framenum in range(count):
      file_path = f'R-{framenum:003}-cd.dat'
      with open(file_path, 'r') as f:
          raw = np.loadtxt(f, delimiter=',', usecols=2)
      data_polar = raw.reshape(n_r, n_phi)

      img = data_polar[i_r, i_phi]
      img[outside] = 0.0

      video.write(_colorise(img))
  video.release()


def create_velocity_video_cylindrical(count, simulation_name, n_r, n_phi, img_size=None):
  r"""
  Create a video of the velocity magnitude from the cylindrical velocity
  snapshots.

  The magnitude :math:`\lvert \mathbf{v} \rvert` is stored in column index 4 of
  the ``V-*-cd.dat`` files, whose format is
  ``r_μm, phi_rad, vr, v_phi, |v|``. As in
  :func:`create_video_cylindrical`, the :math:`(r, \varphi)` plane is resampled
  onto a square Cartesian image.

  Args:
      count (int): Total number of frames.
      simulation_name (str): Base name for the output file.
      n_r (int): Number of radial grid points, matching the snapshot files.
      n_phi (int): Number of azimuthal grid points, matching the snapshot
          files.
      img_size (int): Side length of the square output image in pixels.
          Defaults to ``2 * n_r``.
  """
  if img_size is None:
      img_size = 2 * n_r

  FPS = 10
  SimulationName = simulation_name + f'_fps{FPS}_frame{count}'
  video = VideoWriter(f'{SimulationName}_velocity.mp4', 0x7634706d, float(FPS), (img_size, img_size))

  i_r, i_phi, outside = _polar_to_cartesian_map(n_r, n_phi, img_size)

  for framenum in range(count):
      file_path = f'V-{framenum:003}-cd.dat'
      with open(file_path, 'r') as f:
          raw = np.loadtxt(f, delimiter=',', usecols=4)
      data_polar = raw.reshape(n_r, n_phi)

      img = data_polar[i_r, i_phi]
      img[outside] = 0.0

      video.write(_colorise(img))
  video.release()


def create_velocity_video(count,\
                 simulation_name,\
                 n1,n3
                 ):
  r"""
  Create a video from the Cartesian velocity snapshots.

  The third column of each ``V-*-cd.dat`` file holds the velocity magnitude on
  the :math:`(n_1, n_3)` plane; every frame is colour-mapped and written to
  ``{simulation_name}_fps10_frame{count}_velocity.mp4``.

  Args:
      count (int): Total number of frames, i.e. of snapshot files to read.
      simulation_name (str): Base name for the output file.
      n1 (int): Number of grid points along x.
      n3 (int): Number of grid points along z.
  """

  FPS=10
  SimulationName = simulation_name + f'_fps{FPS}_frame{count}'

  VideoDims=(n1,n3)
  frames=count
  video=VideoWriter(f'{SimulationName}_velocity.mp4', 0x7634706d, float(FPS), VideoDims)

  for framenum in range(frames):
    file_path = f'V-{framenum:003}-cd.dat'
    file = open(file_path,'r')
    img=np.reshape(np.loadtxt(file, delimiter=',', usecols=2),VideoDims)
    file.close()

    zeroed=img-np.min(img)*np.ones(img.shape)
    intImg= np.uint8((255/np.max(zeroed))*zeroed)

    intImgR = np.where(intImg<230 , intImg, 255 )
    intImgG = np.where((intImg<=127) ,intImg, 255-intImg)
    intImgB = np.where((intImg>40) , 255-intImg, 0)
    img3=np.dstack((intImgB,intImgG,intImgR))

    video.write(img3)
  video.release()
