Aiotize Internship Task
Image Segmentation and Crowd Density Estimator

Instruction for Testing:

    Requirements -
        Numpy
        PyTorch
        CV2
        Matplotlib

    Place the image required for testing in the same directory.
    Rename IMAGE_PATH variable (cell number 3) to name of the image that you want to test.
    Running cell number 5 initiates a download of a trained model. Downoaded file should be in the same directory of the notebook.
    Run all other cells as is, in the same order.

Algorithms and Resources Used

        1) Non-Maximal Suppression:
            Object detection involves figuring out multiple areas of interest and plotting certain 'bounding boxes' around them.
            NMS is a filtering process to select the best 'box' or boundary that detects objects.

        Referenced Articles and Code:
            https://www.analyticsvidhya.com/blog/2020/08/selecting-the-right-bounding-box-using-non-max-suppression-with-implementation/
            https://www.youtube.com/watch?v=YDkjWEN8jNA&t=682s&ab_channel=AladdinPersson
            https://www.youtube.com/watch?v=VAo84c1hQX8&t=277s&ab_channel=DeepLearningAI
        
        2) Intersection-over-Union (IoU):
            NMS uses IoU as an evaluation metric to predict bounding boxes. It computes the ratio between [Area of Overlapping] to [Area of Union], to separate bounding boxes               that overlap with each other (boxes that identify the same object).

        Referenced Articles and Code:
            https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
            https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
            https://www.youtube.com/watch?v=ANIzQ5G-XPE&ab_channel=DeepLearningAI
        
        3) Research Paper and CNN:
            A previously published research paper that is similar to given problem statement was referenced and studied.
            The pretrained CNN model 'DarkNet' was specifically used for object detection. The link to downoad the
            trained model is specified in the notebook. (cell 5)

            Paper -  https://arxiv.org/abs/1612.08242
            Tutorial - https://www.youtube.com/watch?v=n9_XyCGr-MI&t=4s&ab_channel=AladdinPersson

        4) Further References from a previous mini project (Harris Corner Detection implementation):
            
            Colab Code : https://colab.research.google.com/drive/1S6v7arQHbQHgiNU-zns5HgHLfEPTcFAA?usp=sharing
            https://stackoverflow.com/questions/51740695/non-maximum-suppression-in-corner-detection
            https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html
