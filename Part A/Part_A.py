import cv2

face_cascade = cv2.CascadeClassifier("face_detector.xml")

#Load the sample videos
exercise_video = cv2.VideoCapture("exercise.mp4")
office_video = cv2.VideoCapture("office.mp4")
street_video = cv2.VideoCapture("street.mp4")


#Counter to keep track of amount of times the method has been acessed
counter = 0
def build_video(video_sample):
    global counter
    counter += 1
    
    #Load the talking video
    talking_video = cv2.VideoCapture("talking.mp4")
    
    #Extracting minimum total number of frames from both the videos
    total_no_frames= min(int(talking_video.get(cv2.CAP_PROP_FRAME_COUNT)), int(video_sample.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    #Extracting width and height of the sample video
    vid_width = int(video_sample.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(video_sample.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    '''
    - The final processed video. VideoWriter takes in 4 parameters:
         1. File name
         2. Character code of codec used. Eg. VideoWriter::fourcc('P','I','M','1') for MPEG-1 codec
         3. Frames per second
         4. The size of the frames
    
    '''
    out = cv2.VideoWriter(f"video_output{counter}.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30.0, (vid_width, vid_height))
    
    
    #Values for determining the first quadrant of the video sample
    overlay_width = int(vid_width / 4)
    overlay_height = int(vid_height / 4)
    
    #Values for determining the starting position for the talking video to overlay
    starting_x = int(vid_width / 10)
    starting_y = int(vid_height / 10)
    
    #Reading the watermarks in color
    watermark1 = cv2.imread("watermark1.png",1)
    watermark2 = cv2.imread("watermark2.png",1)
    
    #Resizing the watermarks according to the specified video frames from the sample video
    watermark1 = cv2.resize(watermark1, (vid_width,vid_height))
    watermark2 = cv2.resize(watermark2, (vid_width,vid_height))
    
    #Setting an interval counter for the watermarks to appear
    interval = int(total_no_frames/10)
    
    '''
    Looping through the minimum total number of frames from talking video and video sample to:
        1. Read and extract all the frames in the talking video and save it to frame variable.
        2. Apply a border of 15 pixels on top, bottom, left right, with constant color of black.
        3. Read and extract all the frames in the video sample and save it to  frame2 variable.
        4. Resize the talking video frames with values overlay width and overlay height specified above.
        5. Detect faces in frame2 (video sample) and loop it to apply Gaussian Blur.
        6. Overlay frame (talking_video) on frame2 (video_sample) positions.
        7. Apply watermarks by using addWeighted that blends frame2(now overlayed with talking + faces blurred) 
            and watermark together.
    '''
    for i in range(0, total_no_frames):
        ret, frame = talking_video.read()
    
        talking_borded = cv2.copyMakeBorder(frame, 15, 15, 15,15, cv2.BORDER_CONSTANT)
    
        ret2, frame2 = video_sample.read()
        
        frame = cv2.resize(talking_borded, (overlay_width, overlay_height))
        
        '''
        - In order to find faces or eyes we use detectMultiScale(), It takes in 3 paramters:
            1. Frame
            2. scaleFactor: Specifies how much the image size is reduced at each image scale.
            3. minNeigher: Specifies how many neighbors each candidate rectangle should have to retain it.
            
        - Once faces are detected it returns the positions of detected faces as rectangles with(x,y,w,h)
        
        - We loop each of this returned face positions to apply GaussianBlur in each frame
        '''
        detections = face_cascade.detectMultiScale(frame2,scaleFactor=1.0485258, minNeighbors=5)
        for face in detections:
            x,y,w,h = face
            
            frame2[y:y+h, x:x+w] = cv2.GaussianBlur(frame2[y:y+h, x:x+w], (15, 15), 0)
            
            
        #Place the talking video into the specified positions of the video sample
        frame2[starting_y:starting_y + overlay_height, starting_x:starting_x + overlay_width] = frame
        
        '''
        - Adds the frame in frame2 and watermarks. addWeighted has 5 parameters:
            1. Frame.(source 1)
            2. Alpha value, sets the transparency level of the frame (0-1).
            3. Watermark image.(source 2)
            4. Beta value, sets the transparency level of the watermark (0-1).
            5. Gamma value, sets the the brightness level
        '''
        final_frame = cv2.addWeighted(frame2, 1, watermark1, 0.7,0)
        
        #Using the interval to switch from watermark1 to watermark2
        if (i >= 2*interval and i <= 4*interval ) or (i >= 6*interval and i <= 8*interval):
            final_frame = cv2.addWeighted(frame2, 1, watermark2, 0.7,0)
    
        #Finally, writing the finalized frames into the output
        out.write(final_frame)
        
    talking_video.release()
    out.release()
    cv2.destroyAllWindows()

build_video(street_video)
build_video(office_video)
build_video(exercise_video)