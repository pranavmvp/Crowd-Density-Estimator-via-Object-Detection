Aiotize Internship Task - 
Image Segmentation and Crowd Density Estimator

Instruction for Testing:

	Requirements -
        	Numpy
        	PyTorch
        	CV2
        	Matplotlib
		Flask

	Testing for Jupyter Notebook:
		1)Place the image required for testing in the same folder.
    	2)Rename IMAGE_PATH variable (cell number 3) to name of the image that you want to test.
     	3)Running cell number 5 initiates a download of a trained model (pre-trained weights). Downoaded file should be in the same directory of the notebook.
    	4)Run all other cells as is, in the same order.

	Testing for API:
		1)Make sure pre-trained weights are available.
	    	2)Execute app.py from terminal
	    	3)View the page at localhost:5000 or http://127.0.0.1:5000/ in your browser.
		4)Choose your input image (from anywhere on your PC), and click submit.
	    	5)A copy of the input image, and the output are stored directly in the same directory.
    

Algorithms and Resources Used:
    
    1) Non-Maximal Suppression:
        Object detection involves figuring out multiple areas of interest and plotting certain 'bounding boxes' around them.
        NMS is a filtering process to select the best 'box' or boundary that detects objects.
	
    2) Intersection-over-Union (IoU):
        NMS uses IoU as an evaluation metric to predict bounding boxes. It computes the ratio between [Area of Overlapping] to [Area of Union], to separate bounding boxes that 	overlap with each other (boxes that identify the same object).

    3) Research Paper and CNN:
        A previously published research paper that is similar to given problem statement was referenced and studied.
        The pretrained CNN model 'DarkNet' was used for training the data.

        Paper -  https://arxiv.org/abs/1612.08242
        
    4) Further References from a previous mini project (Harris Corner Detection implementation):
        Colab Code : https://colab.research.google.com/drive/1S6v7arQHbQHgiNU-zns5HgHLfEPTcFAA?usp=sharing

        
