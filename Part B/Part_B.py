import cv2
import numpy as np

def extract_img(img_file):
    '''
    This function is used to summarise the whole extraction process so that it can be called upon multiple times.
    It also has the image file name as a parameter.

    '''
    
    # Read the image and convert to grayscale
    img = cv2.imread(img_file, 1)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use otsu's threshold to determine an optimal threshold value 
    # cv2.THRESH_BINARY_INV inverts the image where the value 0 is applied to pixels that have the value less than or equal to threshold
    # and the value 255 applied to pixels that have values more than the threshold
    # First variable stores the threshold value
    # Second variable stores the converted image
    threshold_value, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Detect if there is any table in the image and remove it
    # Using a vertical kernel and cv2.erode to obtain the vertical lines
    vertical_kernel = np.ones((15, 1), dtype = np.uint8)
    vertical_erode = cv2.erode(binary_img, vertical_kernel, iterations = 4)
    # Apply cv2.dilate to restore some of the eroded parts of the lines
    vertical_dilate = cv2.dilate(vertical_erode, vertical_kernel, iterations = 4)

    # Using a horizontal kernel and cv2.erode to obtain the horizontal lines
    horizontal_kernel = np.ones((1, 15), dtype = np.uint8)
    horizontal_erode = cv2.erode(binary_img, horizontal_kernel, iterations = 4)
    # Apply cv2.dilate to restore some of the eroded parts of the lines
    horizontal_dilate = cv2.dilate(horizontal_erode, horizontal_kernel, iterations = 4)
    
    # Combine the vertical lines image and horizontal lines image
    combined_eroded_img = cv2.add(vertical_dilate, horizontal_dilate)
    
    # Use cv2.findContours() to get the coordinates of the boundaries of the table
    table_contours, table_hierachy = cv2.findContours(combined_eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Use the values obtained from cv2.boundingRect() to filter out the table
    # There are 4 values returned: 
    # 1) x-coordinate of the top left starting point of the rectangle
    # 2) y-coordinate of the top left starting point of the rectangle 
    # 3) Width of the rectangle
    # 4) Height of the rectangle
    table_perimeter = [cv2.boundingRect(contour) for contour in table_contours]
    
    # Remove the tables
    for x,y,w,h in table_perimeter:
        binary_img[y:y+h, x:x+w] = 0
        
    # Now the table has been removed (if any), move on to identifying each paragraph
    
    # Dilate the adjacent words in a paragraph to form paragraph blocks
    kernel = np.ones((5,5), dtype = np.uint8)
    dilated_img = cv2.dilate(binary_img, kernel, iterations=7)
    
    # Find the contours of all blocks of paragraphs 
    contours, hierachy = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the values of the bouding rectangle of each paragraph
    bounding_rect = [cv2.boundingRect(contour) for contour in contours]
    
    # for c in contours:
    #     x,y,w,h = cv2.boundingRect(c)
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
    
    
    # Sorting the paragraphs
    # Initialise an empty list for each column
    first_column = []
    second_column = []
    third_column = []
    
    # Calculate the average width of a paragraph in the image
    avg_prg_width = sum([i[2] for i in bounding_rect])/len(bounding_rect)
    
    def split_paragraphs(bounding_rect_list):
        ''' 
        This function is used to identify which column each paragraph belongs to.
        It takes a list of values obtained using cv2.boundingRect() as a parameter.
        After sorting, the bounding rectangle's value of each paragraph is appended
        to the specified column's list.
        '''
        
        for i in range(len(bounding_rect_list)):
            # First column (x coordinate < width of paragraph)
            if bounding_rect_list[i][0] <= avg_prg_width:
                first_column.append(bounding_rect_list[i])
                
            # Second column (x coordinate > width of 1 paragraph but <= width of 2 paragraph)
            elif (bounding_rect_list[i][0] > avg_prg_width) and (bounding_rect_list[i][0] <= avg_prg_width*2):
                second_column.append(bounding_rect_list[i])
                
            # Third column (x coordinate > value of width of 2 paragraph)
            elif bounding_rect_list[i][0] > avg_prg_width*2:
                third_column.append(bounding_rect_list[i])
            
    def sort_paragraphs():
        '''
        This function is used to sort the paragraphs into their respective column.
        It then combines the paragraphs into a list and returns the list as the output
        '''
        
        # Check if there is 2 or 3 column before executing
        # The paragraph is sorted by the y-coordinate of the paragraph's bounding rectangle
        # from low to high as the top of the page starts of at y=0
        if first_column is not None:
            first_column.sort(key = lambda x: x[1])
        
        if second_column is not None:
            second_column.sort(key = lambda x: x[1])
        
        if third_column is not None:
            third_column.sort(key = lambda x: x[1])
            
        # Combine the paragraphs into a list and return it
        sorted_prgh = first_column + second_column + third_column
        return sorted_prgh
        
    # Split the paragraphs into their respective columns 
    split_paragraphs(bounding_rect)
    
    # Sort the paragraph based on the correct sequence 
    # The sequence starts from first column going from top to bottom, then moving on 
    # to the next column
    sorted_paragraphs = sort_paragraphs()
    
    # Crop the image and save it into it's own jpg file
    for p in range(len(sorted_paragraphs)):
          x,y,w,h = sorted_paragraphs[p][0], sorted_paragraphs[p][1], sorted_paragraphs[p][2], sorted_paragraphs[p][3]
          cv2.imwrite(f'{img_file}_Paragraph_{p+1}.jpg', img[y:y+h,x:x+w])

# Using a for loop to automate the process of extracting the paragraphs from the images
for i in range(0,8):
    extract_img(f"00{i+1}.png")
