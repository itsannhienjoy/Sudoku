import cv2
import numpy as np
import gradio as gr
import torch
import tensorflow as tf
from fastai.vision.all import *
from itertools import combinations
import copy

from sys import platform
if platform == "win32":
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath
path = Path()

# Fix for torch 1.13
if ismin_torch('1.13') and notmax_torch('1.14'):
    from torch.overrides import has_torch_function_unary, handle_torch_function
    @patch
    def __format__(self:Tensor, format_spec):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__format__, (self,), self, format_spec)
        if self.dim() == 0 and not self.is_meta and issubclass(type(self), Tensor):
            return self.item().__format__(format_spec)
        return object.__format__(self, format_spec)

learn_inf = load_learner(path/'digit_classifier.pkl')
possible_numbers = set(list('123456789')) 
key_squares = [(0,0), (1,3), (2,6), (3,1), (4,4), (5,7), (6,2), (7,5), (8,8)] 

def euclidian_distance(point1, point2):
    '''Calcuates the euclidian distance between the point1 and point2
    used to calculate the length of the four sides of the square'''
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def order_corner_points(corners):
    '''The points obtained from contours may not be in order because of the skewness  of the image, or
    because of the camera angle. This function returns a list of corners in the right order'''
    sort_corners = [[corner[0][0], corner[0][1]] for corner in corners]
    x, y = [], []

    for _,(i,j) in enumerate(sort_corners):
        x.append(i)
        y.append(j)
    centroid = [sum(x) / len(x), sum(y) / len(y)]

    for _, item in enumerate(sort_corners):
        if item[0] < centroid[0]:
            if item[1] < centroid[1]:
                top_left = item
            else:
                bottom_left = item
        elif item[0] > centroid[0]:
            if item[1] < centroid[1]:
                top_right = item
            else:
                bottom_right = item
    ordered_corners = [top_left, top_right, bottom_right, bottom_left]
    return np.array(ordered_corners, dtype="float32")

def image_preprocessing(image, corners):
    '''This function undertakes all the preprocessing of the image'''
    ordered_corners = order_corner_points(corners)
    top_left, top_right, bottom_right, bottom_left = ordered_corners

    width1 = euclidian_distance(bottom_right, bottom_left)
    width2 = euclidian_distance(top_right, top_left)
    width = max(int(width1), int(width2))
    
    # Because a sudoku is a square, you techically only need one dimension. This also seemed to
    # yield better results for the OCR for unknown reason

    # To find the matrix for warp perspective function we need dimensions and matrix parameters
    dimensions = np.array([[0, 0], [width, 0], [width, width],
                           [0, width]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)
    transformed_image = cv2.warpPerspective(image, matrix, (width, width))
    transformed_image = cv2.resize(transformed_image, (400, 400), interpolation=cv2.INTER_AREA)
    return transformed_image

def get_square_box_from_image(image):
    ''' This function returns the top-down view of the puzzle in grayscale.'''
    global big
    if image.shape[1] > 1000: big = True
    else: big = False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 80: # Detect if the grid has a dark background. It most likely has a white grid, so invert it.
        gray = cv2.threshold(gray,gray.mean(),255,cv2.THRESH_BINARY_INV)[1]
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    adaptive_threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    corners = cv2.findContours(adaptive_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corners = corners[0] if len(corners) == 2 else corners[1]
    corners = sorted(corners, key=cv2.contourArea, reverse=True)
    corner = corners[0]
    length = cv2.arcLength(corner, True)
    approx = cv2.approxPolyDP(corner, 0.015 * length, True)
    puzzle_image = image_preprocessing(image, approx)

    return puzzle_image

def get_hough_lines(pic):
    '''Detect all the horizontal and vertical lines in the grid.
    If you do not detect 10 of each, report an error. Collect the coordinates of each
    to define the cell coordinates'''
    global answer
    answer = pic
    gray = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
    if gray.mean() < 80: # If the grid has a dark background (most likely white lines), invert it.
        gray = cv2.threshold(gray,gray.mean(),255,cv2.THRESH_BINARY_INV)[1]
        blur = cv2.GaussianBlur(gray,(5,5),0)
    else:
        blur = cv2.GaussianBlur(gray, (5,5), 0)
    adaptive_threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7,2)
    edges = cv2.Canny(adaptive_threshold,140,255,apertureSize = 3)

    limit = 210
    filtered_lines = []

    while len(filtered_lines) < 20 and limit > 0: 
        # Because different images will have a different threshold limit to detect the lines, lower it until it is low enough.
        # Starting too low isn't good because you'll get too much noise
        limit -= 10
        lines = cv2.HoughLines(edges,1,np.pi/180,limit)

        if not lines.any():
            print('No lines were found');exit()

        theta_threshold = 0.05
        rho_threshold = 30 
        # Given a perfectly regular grid, no cell should be larger than 44 pixels. 30 is sufficient margin of error to not detect lines in the middle of cells in the case of noise like Moire

        # How many lines are similar to a given one
        similar_lines = {i : [] for i in range(len(lines))}
        for i in range(len(lines)):
            for j in range(len(lines)): # You might think this is not necessary. Why not range(i,len(lines)) ? This messes up the number of similar lines. See below
                if i == j:
                    continue
                rho_i,theta_i = lines[i][0]
                rho_j,theta_j = lines[j][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    similar_lines[i].append(j)

        # ordering the indices of the lines by how many are similar to them
        indices = [i for i in range(len(lines))]
        indices.sort(key=lambda x : len(similar_lines[x]))

        line_flags = len(lines)*[True]
        for i in range(len(lines) - 1):
            if not line_flags[indices[i]]: # If we already disregarded the ith element in the ordered list then we don't care
                continue
            rho_i,theta_i = lines[indices[i]][0]
            
            if 0.02 < theta_i < 1.54 or theta_i > 1.59: # Discard lines that aren't horizontal or vertical
                line_flags[indices[i]] = False
                for j in similar_lines[indices[i]]:
                    line_flags[j] = False

            else:
                for j in similar_lines[indices[i]]:
                    if indices.index(j) > i: # Only consider the elements that had less similar lines. This way, you save the line at the "center" of the blob
                        line_flags[j] = False             

        filtered_lines = []

        for i in range(len(lines)): 
            if line_flags[i]:
                filtered_lines.append(lines[i])

    vertical_lines, horizontal_lines = [], []

    for line in filtered_lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        if theta < 0.02:
            vertical_lines.append((x1+x2)//2)
            # cv2.line(pic,(x1,y1),(x2,y2),(0,0,255),2)
        else:
            horizontal_lines.append((y1+y2)//2)    
            # cv2.line(pic,(x1,y1),(x2,y2),(0,0,255),2)

    # cv2.imwrite('hough.jpg',pic)
    horizontal_lines = sorted(horizontal_lines)
    vertical_lines = sorted(vertical_lines)
    
    return [horizontal_lines, vertical_lines, gray]

def recognize_digits(hough_output):
    '''Detect squares that have a number in them, do additionnal image processing and send it to Tesseract for OCR
    Return a puzzle string and the list of the coordinates for where to write a number in each cell'''
    indexes = {}
    n = -1
    horizontal_lines, vertical_lines, gray = hough_output
    if len(horizontal_lines) != 10 or len(vertical_lines) != 10:
        puzzle = input('Failed to properly detect the lines. Enter the puzzle manually, row after row, with 0 or . for blanks')
        return puzzle, indexes

    puzzle = ''

    for j,x in enumerate(horizontal_lines[:-1]):
        x1 = horizontal_lines[j+1]
        for i,y in enumerate(vertical_lines[:-1]):
            n += 1
            y1 = vertical_lines[i+1]
            square = gray[x:x1, y:y1]
            center = square[10:-10,10:-10]

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
            K = 2
            # Do color quantization on the center, aiming for 2 colours maximum.
            Z = center.reshape((-1,1))
            Z = np.float32(Z)
            _, label, c = cv2.kmeans(Z,K,None,criteria,4,cv2.KMEANS_RANDOM_CENTERS)
            c = np.uint8(c)      
            res = c[label.flatten()]
            center = res.reshape((center.shape))
    
            text = ''

            if c.max() - c.min() > 20: # Two colours in the center, so there's a digit
                blur = cv2.GaussianBlur(square,(5,5),0)
                oh,ow = square.shape[:2]
                
                if center[0][0] == c.min(): # Should handle white grid with colored cells with white number in them
                    square = cv2.threshold(square,center.mean()+10,255,cv2.THRESH_BINARY_INV)[1]
                    blur = square.copy()
                
                thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
                contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]
                for cont in contours:
                    x2,y2,w,h = cv2.boundingRect(cont)
                    if (ow//7 <= w < ow-3) and (oh//3 <= h < oh-3):
                        square = square[y2:y2+h,x2:x2+w]
                        # cv2.rectangle(thresh, (x2, y2), (x2 + w, y2 + h), (255,255,255), 2) 
                        break
                else:
                    square = square[8:-8,8:-8]
                fill = center.max()

                # Create new image for OCR to parse                    
                height,width = square.shape[:2]
                nh, nw = height+2, width+10
                blank = fill*np.ones(shape = (nh,nw), dtype=np.uint8)
                height_offset = (nh-height)//2 +1
                width_offset = (nw-width)//2 
                blank[height_offset:height_offset+height,width_offset:width_offset+width] = square
                square = blank
                scale = 56/nh
                square = cv2.resize(square, (int(scale*nh),int(scale*nw)), interpolation=cv2.INTER_LINEAR)
                
                # Do color quantization on it to get rid of Moire artifacts or similar
                # Z = square.reshape((-1,1))
                # Z = np.float32(Z)
                # ret, label, c = cv2.kmeans(Z,K,None,criteria,4,cv2.KMEANS_RANDOM_CENTERS)
                # c = np.uint8(c)
                # res = c[label.flatten()]
                # square = res.reshape((square.shape))
                # global big
                if big:
                    square = cv2.GaussianBlur(square,(5,5),0) # This blur kernel seemed to give the best results on pictures, which had a bigger resolution.
                else:
                    square = cv2.GaussianBlur(square,(3,3),0)
                text = learn_inf.predict(square)[0]
            else:
                text = '.'
                indexes[n] = (x,y)
            
            puzzle += text.replace('\n','')

    return puzzle, indexes

def grid(puzzle):
    """Returns a 2D array represnting the grid with '.' for blanks"""
    sudoku_grid = []
    line = []
    for i,c in enumerate(puzzle.replace(' ','').replace('0', '.')):
        line.append(c)
        if i%9 == 8:
            sudoku_grid.append(list(line))
            line = []
    return sudoku_grid

def get_influencers(y, x):
    """Returns a list of coordinates of all squares directly involved with the considered square"""
    influencers = []
    for f in get_row_neighbours, get_column_neighbours, get_square_neighbours: 
        influencers += f(y, x)
    return influencers

def get_row_neighbours(y, x, width = 9):
    """Returns a list of coordinates for the other squares in a row"""
    return [(y,i) for i in range(width) if i != x]

def get_column_neighbours(y, x, height = 9):
    """Returns a list of coordinates for the other squares in a column"""
    return [(j,x) for j in range(height) if j != y]

def get_square_neighbours(y, x, height = 9, width = 9):
    """Returns a list of coordinates for the other squares in a small square"""
    x_range, y_range = (x // 3) * 3 , (y // 3) * 3 
    return [(j+y_range, i+x_range) for j in range(3) for i in range(3) if not (i+x_range == x and j+y_range == y)]

def get_regions():
    """Return a list of all rows, columns, 3x3 squares
    as well as a list of just the 3x3 squares"""
    regions, squares = [], []
    for (y,x) in key_squares:
        for f in get_row_neighbours, get_column_neighbours, get_square_neighbours:
            region = f(y,x) + [(y,x)]
            regions.append(region)
    return regions

all_regions = get_regions()

def initialize(puzzle, height, width, candidates):
    """Write down every candidates for every single square on the board
    If there's only one candidate, fill that square with the number"""
    for y in range(height):
        for x in range(width):
            if puzzle[y][x] == '.':
                candidates[(y, x)] = possible_numbers.copy()
    return candidates

def single_candidate(puzzle, candidates, changed = False):
    """Write down every candidates for every single square on the board
    If there's only one candidate, fill that square with the number"""
    deletion_list = []
    for (y,x) in candidates:
        influencers = set([puzzle[j][i] for (j,i) in get_influencers(y,x) if puzzle[j][i] != '.'])
        candidates[(y,x)] -= influencers
        if len(candidates[(y,x)]) == 1:
            puzzle[y][x] = list(candidates[(y,x)])[0]
            changed = True; deletion_list.append((y, x))
    for square in deletion_list: del candidates[square]
    return puzzle, candidates, changed

def single_position(puzzle, candidates, changed = False):
    """Within a row, column or 3x3 square, look for a number that has only one candidate.
    Fill the square that has this candidate with the number"""
    for region in all_regions:
        found = set([puzzle[y][x] for (y, x) in region if puzzle[y][x] != '.'])
        searching = possible_numbers - found
        region_candidates = [candidates[(y, x)] for (y, x) in region if puzzle[y][x] == '.']
        for number in searching:
            if sum([list(region_candidate).count(number) for region_candidate in region_candidates])  == 1:
                for (y, x) in region:
                    if puzzle[y][x] == '.' and number in candidates[(y, x)]:
                        puzzle[y][x] = number
                        changed = True; del candidates[(y, x)]
    return puzzle, candidates, changed

def naked_groups(candidates, changed = False):
    """In the case of naked groups, remove candidates from squares
    that aren't in the group"""
    for region in all_regions:
        coords = [(y, x) for (y, x) in region if (y, x) in candidates]
        for i in [2,3]:
            n_uples = [set(p) for p in combinations(coords, i)]
            for n_uple in n_uples:
                numbers = set()
                for coord in n_uple:
                    numbers |= candidates[coord]
                if len(numbers) == i:
                    for (y, x) in coords:
                        if (y, x) not in n_uple and (candidates[(y, x)] - numbers) != candidates[(y, x)]:
                            candidates[(y, x)] -= numbers; changed = True
    return candidates, changed         

def is_solved(puzzle):
    """Return whether a grid is solved"""
    return all([set(''.join([puzzle[y][x] for (y,x) in region])) == possible_numbers for region in all_regions])

def solve(puzzle, candidates, height = 9, width = 9, changed = True, guessing = 0):
    candidates = initialize(puzzle, height, width, candidates)
    while changed:
        changed = False
        puzzle, candidates, changed = single_candidate(puzzle, candidates, changed)
        puzzle, candidates, changed = single_position(puzzle, candidates, changed)
        if not changed: candidates, changed = naked_groups(candidates, changed)

    if not is_solved(puzzle): 
        if any(len(candidates[(y, x)]) == 0 for (y, x) in candidates) or not candidates:
            if not guessing: print('This puzzle cannot be solved')
            else: return False
        choose = min(candidates.items(), key = lambda x:len(x[1]))
        y, x, choices = choose[0][0], choose[0][1], list(choose[1])
        for choice in choices: 
            puzzle_copy, changed_copy = copy.deepcopy(puzzle), True
            puzzle_copy[y][x] = choice
            puzzle_copy = solve(puzzle_copy, {}, height, width, changed_copy, guessing+1)
            if puzzle_copy: 
                if is_solved(puzzle_copy): 
                    return puzzle_copy

    return puzzle

def show_results(puzzle, indexes):
    if not puzzle or not indexes: 
        return False
    else:
        for cell,(x,y) in indexes.items():
            j,i = cell//9, cell%9
            digit_colour = [x[0] for x in cv2.bitwise_not(answer[y,x]).tolist()]
            digit_colour = (0,255,0) if (digit_colour[0] < 50 and digit_colour[1] < 50 and digit_colour[2] > 200) else (0,0,255)
            cv2.putText(answer,puzzle[j][i],(y+13,x+35),cv2.FONT_HERSHEY_SIMPLEX,1,digit_colour,1,cv2.LINE_AA)
        return answer

height,width = 9,9

def parse_and_solve(image):
    image = PILImage.create(image)
    image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
    puzzle, indexes = recognize_digits(get_hough_lines(get_square_box_from_image(image)))
    puzzle = solve(grid(puzzle),{})
    return show_results(puzzle, indexes)

title = "Automatic sudoku solver"
description = "Sudoku puzzle solver from image using openCV and a digit recognizer model fine-tuned using CNN on MNIST dataset."
examples = ["example 1.jpeg", "example 3.jpeg", "easy.png", "sudoku.jpeg"]

intf = gr.Interface(fn=parse_and_solve, inputs=gr.Image(), title = title, description = description, outputs=gr.Image(), examples = examples, cache_examples=False)
intf.launch(inline=False)