"""
	Description: Python script to rename images in folders
"""

import os 
count = 0 
for filename in os.listdir('../MCF12A'):

	if filename.endswith(".png") or filename.endswith(".jpg"): 
		count += 1
		os.rename('../MCF12A/'+filename, '../MCF12A/'+'MCF12A('+str(count)+').jpg')

