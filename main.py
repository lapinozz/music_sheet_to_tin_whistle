import os
from pathlib import Path
import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random as rng
import musicalbeeps

import pygame
import time

from itertools import groupby

class NoteType:
	Note = 1
	Rest = 2

	Flat = -1

class ClefType:
	G = 1

class dotdict(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

def trimRect(frame):
	f2 = np.array(frame)
	r = np.nonzero(f2)
	y_nonzero = r[0]
	x_nonzero = r[1]
	return dotdict({"top": np.min(y_nonzero), "bottom": np.max(y_nonzero), "left": np.min(x_nonzero), "right": np.max(x_nonzero)})

def trimImg(frame):
	rect = trimRect(frame)
	return frame[rect.top:rect.bottom, rect.left:rect.right]

def drawText(img, text, align, pos, font, fontScale, color, thickness, lineType):
	(textW, textH), baseline = cv.getTextSize(text, font, fontScale, thickness)

	if align[0] == 0:
		pos = (pos[0] - textW/2, pos[1])
	elif align[0] > 0:
		pos = (pos[0] - textW, pos[1])

	if align[1] == 0:
		pos = (pos[0], pos[1] + textH/2)
	elif align[1] < 0:
		pos = (pos[0], pos[1] + textH)

	pos = (int(pos[0]), int(pos[1]))

	cv.putText(img, text, pos, font, fontScale, color, thickness, lineType, False)

	return dotdict({"top": pos[1] - textH, "bottom": pos[1], "left": pos[0], "right": pos[0] + textW})

def show_wait_destroy(winname, img):
	cv.imshow(winname, img)
	cv.moveWindow(winname, 500, 0)
	cv.waitKey(0)
	cv.destroyWindow(winname)

def main(argv):
	if len(argv) < 1:
		print ('Not enough parameters')
		return -1

	src = cv.imread(argv[0], cv.IMREAD_COLOR)

	if src is None:
		print ('Error opening image: ' + argv[0])
		return -1

	#cv.imshow("src", src)

	#show_wait_destroy("gray", gray)

	sheetWithInfo = cv.resize(src, (0,0), fx=1, fy=1)
	sheetWithInfo = 255 - trimImg(255 - sheetWithInfo)
	#show_wait_destroy("r", sheetWithInfo)

	if len(src.shape) != 2:
		gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
	else:
		gray = src

	gray = cv.bitwise_not(gray)
	bw = cv.adaptiveThreshold(gray, 255, cv.THRESH_BINARY, cv.THRESH_BINARY, 9, -31)
	ret3,bw = cv.threshold(bw,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

	rawBw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY)[1]

	linesRect = trimRect(rawBw)
	rowSums = np.sum(trimImg(rawBw),axis=1).tolist()
	sumMax = np.max(rowSums)
	rowSumsNormalized = rowSums / sumMax

	peaksThreshold = 0.7
	peaks = rowSumsNormalized > peaksThreshold

	peakIndices = np.where(peaks == 1)[0] + linesRect.top

	linesImg = cv.cvtColor(rawBw, cv.COLOR_GRAY2BGR)
	for i in range(peakIndices.size):
		index = peakIndices[i]
		linesImg[index:index+1,:] = (0,255,0)

	staffs = []
	lines = []

	start = -1;
	diff = -1;
	count = 0;
	for i in range(peakIndices.size):
		index = peakIndices[i]

		if i > 0 and peakIndices[i - 1] + 1 == peakIndices[i]:
			continue

		lines.append(index)

		count += 1

		if count == 2:
			diff = index - start
		elif count > 1:
			d = index - peakIndices[i - 1]
			r = d / diff
			
			if i > 0 and (r < 0.8 or r > 1.2):
				count = 1
				start = -1
				lines = [index]
				linesImg[index:index+1,:] = (0,0,255)
			elif count == 5:
				staffs.append(dotdict({'startRow': start, 'endRow': index, 'lines': lines}))
				count = 0
				start = -1
				lines = []

		if start < 0 and count == 1:
			start = index

	Path(f'./debug/').mkdir(parents=True, exist_ok=True)

	#show_wait_destroy("lines_detection", linesImg)
	cv.imwrite(f'./debug/lines_detection.png', linesImg)

	noteTemplate = cv.imread('./templates/note-template.png',0)
	doubleFlagTemplate = cv.imread('./templates/double-flag-template.png',0)
	clefTemplateG = cv.imread('./templates/clef-template-g.png',0)
	flatTemplate = cv.imread('./templates/flat-template.png',0)
	halfBeatRestTemplate = cv.imread('./templates/half-beat-rest-template.png',0)
	quarterBeatRestTemplate = cv.imread('./templates/quarter-beat-rest-template.png',0)

	staffIndex = 0
	for staff in staffs:
		staffIndex = staffIndex + 1
		staff.index = staffIndex

		Path(f'./debug/staff_{staffIndex}').mkdir(parents=True, exist_ok=True)

		startRow = staff['startRow']
		endRow = staff['endRow']
		lines = staff['lines']

		height = int((endRow - startRow) / 1.5)

		lineColumns = np.where((rawBw[lines, :] > 0).all(axis=0))[0]

		xStart = lineColumns[0]
		xEnd = lineColumns[-1]

		yStart = startRow - height
		yEnd = endRow + height

		staffImg = bw[yStart:yEnd, xStart:xEnd]
		staffImgOriginal = src[yStart:yEnd, xStart:xEnd].copy()

		staffImg[0:8, 0:8] = 0
		staffImgOriginal[0:8, 0:8] = 255

		lines = lines - yStart
		staff.localLines = lines

		verticalsize = 3
		verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
		staffImg = cv.erode(staffImg, verticalStructure)

		verticalsize = 3
		verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (2, verticalsize))
		staffImg = cv.dilate(staffImg, verticalStructure)

		verticalsize = 2
		verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (2, verticalsize))
		staffImgInflated = cv.dilate(staffImg, verticalStructure)
				
		templateDrawing = cv.cvtColor(staffImg.copy(), cv.COLOR_GRAY2BGR) #np.zeros((staffImgInflated.shape[0], staffImgInflated.shape[1], 3), dtype=np.uint8)

		# Find Notes
		notes = []
		def findNotes(template, thresh, type):
			localNotes = []
			w, h = template.shape[::-1]
			res = cv.matchTemplate(staffImgInflated, template, cv.TM_CCOEFF_NORMED)
			notePositions = zip(*np.where( res >= thresh)[::-1])
			for pt in notePositions:
				note = dotdict({"x": pt[0], "y": pt[1], "w": w, "h": h})
				note.left = note.x
				note.top = note.y
				note.right = note.x + note.w
				note.bottom = note.y + note.h
				note.midX = note.x + note.w / 2
				note.midY = note.y + note.h / 2
				note.linePos = note.midY - 1
				
				note.type = type
				note.dots = 0
				note.flags = 0
				note.beams = 0
				
				localNotes.append(note)
			return localNotes			

		for note in findNotes(noteTemplate, 0.7, NoteType.Note):
			notes.append(note)
		
		for note in findNotes(halfBeatRestTemplate, 0.9, NoteType.Rest):
			note.beats = 1/2
			notes.append(note)
		
		for note in findNotes(quarterBeatRestTemplate, 0.9, NoteType.Rest):
			note.beats = 1/4
			notes.append(note)

		flats = []
		for flat in findNotes(flatTemplate, 0.9, NoteType.Flat):
			flats.append(flat)

		# Remove duplicate notes
		x = 0
		while x < len(notes):
			y = x + 1
			while y < len(notes):
				n1 = notes[x]
				n2 = notes[y]

				xOverlap = min(abs(n1.left - n2.left), abs(n1.right - n2.right))
				yOverlap = min(abs(n1.top - n2.top), abs(n1.bottom - n2.bottom))
				overlap = max(xOverlap, yOverlap)

				overlaps = overlap < n1.w / 2

				if (overlaps):
					del notes[y]
				else:
					y += 1
			x += 1

		# Find average distance between lines in the staff
		avgLineDist = 0
		for i in range(len(lines) - 1):
			avgLineDist += lines[i + 1] - lines[i]
		avgLineDist = round(avgLineDist / (len(lines) - 1))

		midLinePos = lines[2]       

		notes.sort(key=lambda n: n.left)   

		# Find Clef
		res = cv.matchTemplate(staffImg, clefTemplateG, cv.TM_CCORR_NORMED)
		min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
		top_left = max_loc
		if max_val > 0.8:
			staff.clefRect = dotdict({"top": max_loc[1], "bottom": max_loc[1] + clefTemplateG.shape[0], "left": max_loc[0], "right": max_loc[0] + clefTemplateG.shape[1]})
			staff.clefType = ClefType.G

		# Find Signature
		staff.signatures = []
		celfEnd = staff.clefRect.right
		firstNoteStart = notes[0].left
		for flat in flats:

			# Only consider flats between clef and first note
			if flat.left < celfEnd or flat.right > firstNoteStart:
				continue

			flat.linePos = flat.bottom - flat.w / 1.7
			flat.subline = round((flat.linePos - midLinePos) / (avgLineDist / 2))
			flat.name = 'b'
			staff.signatures.append(flat)

		# Set pitch/letter/name/etc
		noteIndex = 0
		for note in notes:
			if note.type == NoteType.Note:
				note.staffIndex = staffIndex
				note.localIndex = noteIndex
				noteIndex = noteIndex + 1

				sublineCount = round((note.linePos - midLinePos) / (avgLineDist / 2))

				pitchMap = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
				accidentalMap = {'#': 1, '': 0, 'b': -1, '!': -1}
				
				note.accidental = ''
				for sign in staff.signatures:
					if sign.subline == sublineCount:
						note.accidental += sign.name

				note.letter = chr(ord('A') + (-sublineCount + 1) % 7)
				note.octave = 4 + int((-sublineCount + 6) / 7)
				note.pitch = 12 * (note.octave + 1) + pitchMap[note.letter] + accidentalMap[note.accidental]
				note.name = note.letter + str(note.octave) + note.accidental
				note.playName = note.name
			elif note.type == NoteType.Rest:
				note.letter = 'R'
				note.pitch = -1
				note.name = "RT"
				note.playName = "pause"

		# Find Flags and Beams
		for note in notes:
			if note.type != NoteType.Note:
				continue

			halfWidth = int(note['w'] / 2)
			halfHeight = int(note['h'] / 2)

			absoluteLeft = max(0, int(note['left'] - halfWidth))
			absoluteRight = int(note['right'] + halfWidth)

			noteImg = staffImg[:, absoluteLeft:absoluteRight].copy()     
			noteLeft = note['left'] - absoluteLeft
			noteRight = note['right'] - absoluteLeft

			noteImgOg = staffImgOriginal[:, absoluteLeft:absoluteRight].copy()

			Path(f'./debug/staff_{note.staffIndex}/notes').mkdir(parents=True, exist_ok=True)
			cv.imwrite(f'./debug/staff_{note.staffIndex}/notes/note_{note.localIndex}.png', noteImg)
			cv.imwrite(f'./debug/staff_{note.staffIndex}/notes/note_og_{note.localIndex}.png', noteImgOg)

			note.flags = 0

			# Double flags detecton
			res = cv.matchTemplate(noteImg, doubleFlagTemplate, cv.TM_CCORR_NORMED)
			min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
			top_left = max_loc
			if max_val > 0.9:
				note.flags = 2

			note.dots = 0

			# Find Dots
			contoursDrawing = noteImg.copy()
			contours, _ = cv.findContours(noteImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
			contours_poly = [None]*len(contours)
			boundRect = [None]*len(contours)
			centers = [None]*len(contours)
			radius = [None]*len(contours)
			areas = [None]*len(contours)
			for i, c in enumerate(contours):
				contours_poly[i] = cv.approxPolyDP(c, 1, True)
				boundRect[i] = cv.boundingRect(contours_poly[i])
				centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
				areas[i] = cv.contourArea(c)
			
			for i in range(len(contours)):
				color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))

				area = cv.contourArea(contours[i])

				bounds = boundRect[i]
				boundLeft = bounds[0]
				boundRight = bounds[0] + bounds[2]
				boundTop = bounds[1] + bounds[2]
				boundBottom = bounds[1] + bounds[3]

				if area > 5:
					continue

				if boundLeft < noteRight:
					continue

				if boundRight >= noteImg.shape[1]:
					continue

				if boundBottom < note['top'] or boundTop > note['bottom']:
					continue

				cv.drawContours(contoursDrawing, contours_poly, i, color)
				cv.rectangle(contoursDrawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
				  (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
				#show_wait_destroy('Contours', contoursDrawing)
			
				note.dots = 1

			# Find Beams
			verticalsize = 3
			verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (verticalsize, 1))
			noteImg = cv.erode(noteImg, verticalStructure)

			noteImg[note['top'] - halfHeight:note['bottom'] + halfHeight, :] = 0
			stemSliceUp = noteImg[note['top'] - 3:note['top'] - 1, :]
			stemSliceDown = noteImg[note['bottom'] + 1:note['bottom'] + 3, :]

			stemUpSlicePixels = stemSliceUp.sum()
			stemDownSlicePixels = stemSliceDown.sum()

			isStemUp = stemUpSlicePixels > stemDownSlicePixels
			isStemDown = not isStemUp

			stemPos = note['left'] - absoluteLeft

			leftSlice = noteImg[note['bottom']:-1, stemPos-1:stemPos-0]
			rightSlice = noteImg[note['bottom']:-1, stemPos+6:stemPos+7]

			leftSlice = leftSlice.T[0]
			rightSlice = rightSlice.T[0]

			leftBeams = len([sum(1 for i in g) for k,g in groupby(leftSlice) if k > 0])
			rightBeams = len([sum(1 for i in g) for k,g in groupby(rightSlice) if k > 0])

			note.beams = beamCount = max(leftBeams, rightBeams)

		# Set duration and durationStr
		for note in notes:

			numerator = 1
			denominator = 1

			if note.type == NoteType.Note:
				divisorCount = note.flags if note.flags > 0 else note.beams

				if divisorCount == 1:
					denominator = 2

				elif divisorCount == 2:
					denominator = 4

				elif divisorCount == 3:
					denominator = 8

				elif divisorCount > 3:
					raise ValueError('Too many beams')
			elif note.type == NoteType.Rest:
				denominator = 1 / note.beats

			if note.dots > 0:
				numerator *= 1.5
				frac = numerator % 1
				numerator = numerator / frac
				denominator = denominator / frac

			duration = numerator / denominator
			durationStr = str(int(numerator)) + '|' + str(int(denominator))

			note.duration = duration
			note.durationStr = durationStr

			cv.line(noteImg, (stemPos-1, 0), (stemPos-1, noteImg.shape[0]), (255,255,255), 1)
			cv.line(noteImg, (stemPos+6, 0), (stemPos+6, noteImg.shape[0]), (255,255,255), 1)

		scale = 2
		annotatedImg = staffImgOriginal.copy() #np.zeros((staffImgInflated.shape[0], staffImgInflated.shape[1], 3), dtype=np.uint8)
		annotatedImg = cv.resize(annotatedImg, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)

		for note in notes:
			cv.rectangle(templateDrawing, (note.x, note.y), (note.x + note.w, note.y + note.h), (0,0,255), 1)
			cv.line(annotatedImg, (note.x*scale, int(note.linePos)*scale), ((note.x + note.w)*scale, int(note.linePos)*scale), (0,255,0), 1)

		for sign in staff.signatures:
			cv.rectangle(templateDrawing, (sign.x, sign.y), (sign.x + sign.w, sign.y + sign.h), (0,0,255), 1)
			cv.line(annotatedImg, (sign.x*scale, int(sign.linePos)*scale), ((sign.x + sign.w)*scale, int(sign.linePos)*scale), (0,255,0), 1)
		
		topMargin = 40
		annotatedImg = cv.copyMakeBorder(annotatedImg, topMargin, 0, 0, 0, cv.BORDER_REPLICATE)
		annotatedImg[0:topMargin,:] = 255
		linesTrimRect = trimRect(255 - annotatedImg)
		linesTop = linesTrimRect.top

		for note in notes:
			font = cv.FONT_HERSHEY_PLAIN
			fontScale = 0.4*scale
			thickness = 1

			textRect = drawText(annotatedImg, note.name, (0, 1), (int(note.midX)*scale, linesTop), font, fontScale, (255/1.25,255/1.25,0), thickness, cv.LINE_AA)
			textRect = drawText(annotatedImg, note.durationStr, (0, 1), (int(note.midX)*scale, textRect.top - 5), font, fontScale*0.75, (255/1.25,255/1.25,0), thickness, cv.LINE_AA)

		staff.notes = notes

		staff.annotatedImg = annotatedImg

		cv.imwrite(f'./debug/staff_{staffIndex}/staffImg.png', staffImg)
		cv.imwrite(f'./debug/staff_{staffIndex}/staffImgAnnotated.png', annotatedImg)
		cv.imwrite(f'./debug/staff_{staffIndex}/staffImgInflated.png', staffImgInflated)
		
		#show_wait_destroy("r", staffImg)

	annotatedStaffsImgHeight = sum(s.annotatedImg.shape[0] for s in staffs)
	maxWidth = max(staffs, key = lambda s: s.annotatedImg.shape[1]).annotatedImg.shape[1]
	annotatedStaffsImg = np.zeros((annotatedStaffsImgHeight, maxWidth, 3), np.uint8)
	annotatedStaffsImg[::] = 255

	offset = 0
	for staff in staffs:
		img = staff.annotatedImg
		annotatedStaffsImg[offset:offset+img.shape[0], :img.shape[1]] = img
		offset += img.shape[0]

	cv.imwrite(f'./debug/annotatedStaffs.png', annotatedStaffsImg)
	#show_wait_destroy("r", annotatedStaffsImg)

	# Play the song
	beatTime = 60/114
	player = musicalbeeps.Player(volume = 0.5, mute_output = False)
	if False:
		for staff in staffs:
			for note in staff['notes']:
				player.play_note(note.playName, beatTime * note.duration)

	# Tin Whistle Notes
	tinWhileMapping = {
		"D4": [1, 1, 1, 1, 1, 1, 0],
		"E4": [1, 1, 1, 1, 1, 0, 0],
		"F4": [1, 1, 1, 1, 0, 0, 0],
		"F4#": [1, 1, 1, 1, 0, 0, 0],
		"G4": [1, 1, 1, 0, 0, 0, 0],
		"A4": [1, 1, 0, 0, 0, 0, 0],
		"B4": [1, 0, 0, 0, 0, 0, 0],
		"C5": [0, 1, 1, 0, 0, 0, 0],
		"C5#": [0, 0, 0, 0, 0, 0, 0],
		"D5": [0, 1, 1, 1, 1, 1, 1],
		"E5": [1, 1, 1, 1, 1, 0, 1],
		"F5": [1, 1, 1, 1, 0, 0, 1],
		"F5#": [1, 1, 1, 1, 0, 0, 1],
		"G5": [1, 1, 1, 0, 0, 0, 1],
		"A5": [1, 1, 0, 0, 0, 0, 1],
	}

	holeRadius = 9
	holeSpacing = 3
	noteNameSpacing = 5
	noteNameHeight = 10
	noteDurationHeight = 14
	noteImgWidth = holeRadius * 2 + 4
	noteImgCenter = int(noteImgWidth/2)

	notesStart = noteNameHeight + noteNameSpacing * 2
	durationStart = notesStart + (holeRadius * 2 + holeSpacing) * 7 + 2

	noteImgShape = (notesStart + (holeRadius * 2 + holeSpacing) * 7 + noteDurationHeight + 10, noteImgWidth, 3)

	font = cv.FONT_HERSHEY_PLAIN
	fontScale = 0.4*scale
	thickness = 1
	
	Path(f'./out/').mkdir(parents=True, exist_ok=True)

	staffImgs = []
	for staff in staffs:

		staffImg = np.zeros((noteImgShape[0], noteImgShape[1] * len(staff['notes']), noteImgShape[2]), np.uint8)
		staffImg[::] = 255

		i = -1
		for note in staff['notes']:
			i += 1

			noteImg = np.zeros(noteImgShape, np.uint8)
			noteImg[::] = 255

			if note.type == NoteType.Note:
				tinPlay = tinWhileMapping[note.playName]

				textRect = drawText(noteImg, note.name, (0, -1), (noteImgCenter, noteNameSpacing), font, fontScale, (0,0,0), thickness, cv.LINE_AA)

				for x in range(7):
					filled = tinPlay[x] == 1

					center = (noteImgCenter, notesStart + holeRadius + (holeRadius*2 + holeSpacing) * x)

					if x == 6:
						if filled:
							cv.line(noteImg, (center[0] - holeRadius, center[1]), (center[0] + holeRadius, center[1]), (0,0,0), 2)
							cv.line(noteImg, (center[0], center[1] - holeRadius), (center[0], center[1] + holeRadius), (0,0,0), 2)
					else:
						cv.circle(noteImg, center, holeRadius - (0 if filled else 2), (0,0,0), -1 if filled else 2)
				
				#if first:
				#	#allImgs = img
				#else:
				#	#allImgs = np.concatenate((allImgs, img), axis=1)

			drawText(noteImg, note.durationStr, (0, -1), (noteImgCenter, durationStart), font, fontScale * 0.8, (0,0,0), thickness, cv.LINE_AA)

			staffImg[:, noteImgShape[1] * i:noteImgShape[1] * (i+1)] = noteImg

		cv.imwrite(f'./out/staffs_{i}.png', staffImg)
		staffImgs.append(staffImg)
		#show_wait_destroy("r", staffImg)

	maxWidth = max(staffImgs, key = lambda s: s.shape[1]).shape[1]
	staffsImg = np.zeros((noteImgShape[0] * len(staffImgs), maxWidth, noteImgShape[2]), np.uint8)
	staffsImg[::] = 255

	for i in range(len(staffImgs)):
		staffImg = staffImgs[i]
		staffsImg[noteImgShape[0] * i:noteImgShape[0] * (i + 1), :staffImg.shape[1]] = staffImg

	cv.imwrite(f'./out/staffs_all.png', staffsImg)
	show_wait_destroy("r", staffsImg)

	pygame.mixer.init()
	music = pygame.mixer_music

	recordings = {}
	for key in tinWhileMapping:
		recordings[key] = pygame.mixer.Sound(f"./recordings/{key}.wav")

	if False:
		for key in tinWhileMapping:
			sound = recordings[key]
			duration = 0.25
			#sound.set_pos(0)
			sound.play(0, int(duration * 1000))
			time.sleep(duration)
			sound.stop()
			#player.play_note(key, duration)
			#time.sleep(duration)

	if False:
		for staff in staffs:
			for note in staff['notes']:
				duration = beatTime * note.duration# * 5
				print(note.playName)
				#sound.set_pos(0)
				if note.playName == 'pause':
					time.sleep(duration)
				else:
					sound = recordings[note.playName]
					sound.play(0, int(duration * 1000))
					time.sleep(duration)
					sound.stop()
				#player.play_note(note.playName, duration)
				#time.sleep(duration)

if __name__ == "__main__":
	main(sys.argv[1:])