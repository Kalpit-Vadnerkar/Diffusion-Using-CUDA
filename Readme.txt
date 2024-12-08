To run the program, make sure you have glut working on your palmetto system. (A100 or V100 GPUs are recommended)
navigate to the project directory

$ cd Project

Load the gcc and cuda modules

$ module load gcc cuda

Complie the the source code using the make command

$ make clean
$ make

Run the "diffusion viewer" appilcation using vglrun

$ vglrun ./diffusion_viewer

You will get the output:
Diffusion Viewer Usage Commands:
--------------------------------
Image Processing:
  f - Run GPU forward diffusion
  s - Run GPU forward diffusion with shared memory
  r - Run GPU reverse diffusion
  F - Run CPU forward diffusion
  R - Run CPU reverse diffusion

Image Controls:
  i - Reset image to original pattern
  [ - Decrease image size by 256 pixels (min: 256x256)
  ] - Increase image size by 256 pixels (max: 4080x4080)
  , - Decrease beta value (noise factor)
  . - Increase beta value (noise factor)

Performance Controls:
  - - Decrease threads per block by 4
  = - Increase threads per block by 4

Application Control:
  q - Quit application

Current Settings:
  Image Size: 512x512
  Threads per Block: 16x16
  Beta Value: 0.100

A glut window will also open displaying a checkered image. Use the different commands to perform forward and reverse diffusion.
