#!/usr/bin/env python
# coding: utf-8

# ### Lab 3.2 Image Redaction - Exercise
# Now let's practice what you have learned. 
# 
# We are going to a variation. Rather than redact the face, let's redact the areas within the eyes.  

# ### Redact an image - CPU
# Now lets blur the are within the rectangle to redact the face using the OpenCV GaussianBlur function. We will not identify eyes in this code block, just the face.

# In[38]:


get_ipython().run_cell_magic('time', '', "import numpy as np\nimport cv2\nimport ipywidgets\nfrom IPython.display import display\nimport numpy as np\n\n# Reading an image using OpenCV\nimage = cv2.imread('lena_color.tif')\nimage_widget = ipywidgets.Image(format='jpeg')\n\n# Converting BGR image into a RGB image\nimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\ngray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n\nfpath='./cascades/haarcascade_frontalface_alt.xml'\nepath='./cascades/haarcascade_eye.xml'\nface_detect = cv2.CascadeClassifier(fpath)\neye_detect = cv2.CascadeClassifier(epath)\n\nfaces = face_detect.detectMultiScale(gray, 1.3, 5)\nfor (x,y,w,h) in faces:\n    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)\n    roi_gray = gray[y:y+h, x:x+w]\n    roi_color = image[y:y+h, x:x+w]\n    \n    eyes = eye_detect.detectMultiScale(roi_gray)\n    for (ex,ey,ew,eh) in eyes:\n        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)\n        eyeblur = roi_color[ey:ey+eh, ex:ex+ew]\n        eyeblur = cv2.GaussianBlur(eyeblur, (23, 23), 30)\n        roi_color[ey:ey+eyeblur.shape[0], ex:ex+eyeblur.shape[1]] = eyeblur\n    \n  \n# Display the output\nimage = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)\nimage_widget.value = bytes(cv2.imencode('.jpg', image)[1])\ndisplay(image_widget)")


# ### Redact an image -GPU
# Now lets use the GPU to run the classifier. 
# Copy the code from above into this next code block and modify to use the GPU to identify the face and eyes and redact just the eyes. For this exercise, just enable the CUDA classifier for the face detection, not the eye detection.   
# 

# In[4]:


## Add code here


# In[6]:


import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)

