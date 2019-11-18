# Sudoku-Solver-with-Extended-Reality-using-OpenCV
Unfinished project

I recently saw a project (https://www.youtube.com/watch?v=Ob3MW_DKkNA) of a sudoku solver using a camera to detect the sudoku numbers and put the solution on the screen.
I have been trying to recreate this myself and I'm posting the progress.
A lot of the algorithms (detecting edges and getting perspective of image) are taken from different sytes and are common knowlegde.

The only thing left to do is from the segmentation of the sudoku table into image cells, get the corresponding number. A classic number classification algorithm. I've tried doing it myself using the Google Tesseract image to text but nothing worked. Then I came up with a nice Deep Learning architecture but all dataset I had for digit recognition was the MNIST, so it didn't work eighter...
I need to get some good images of "newspapperish" digit images to accomplish accurate classification. In a near future.

The first one is an image of a sudoku and the other one is the algorithm detecting the sudoku and guesing the right perspective.

[! Alt Text](/sudoku.jpg)

[! Alt Text](/warped_cut_image.png)
