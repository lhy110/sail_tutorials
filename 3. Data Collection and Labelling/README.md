# 3. Data Collection and Labelling
By Nandakumar Thachapilly and Zac Todd

## Data Collection
The efficiency of a machine learning model to predict is defined by the data used in
creating it. A dataset with volume and variety of classes (The targets that needs to be
predicted) will positively impact in creating a robust predictive model.
In general, data being collected can be categorised as qualitative (Words, images,
videos etc) and quantitative (Numerical data). Data collected for an image processing
project falls in the qualitative type and has the following steps to be factored during
collection process.

  * **Identify the focus area and goals:** Prepare list of areas to focus which would ensure the variety of data. For example, while collection road surfaces, identify the routes which would likely have more road deformations. Target the volume of data and based on that plan the kilometres to be covered.
  * **Plan the capture methods:** Identify the list of devices to be used for capturing data. Ensure the required infrastructure is in place to fit these devices. Perform trial runs with each of the devices, verifying the accurate capture, proper storage of the data with the necessary power requirements.
  * **Other Arrangements:** Check if there any approvals or paper work to be completed for the data collection
    * Health and Safety: Healthy and safety approvals are normally required in most of the formal data collection activities. Ensure necessary approvals are taken prior to the collection.
    * Policy/Procedure Approvals: Depending on the type of data being collection, verify if there any additional approvals are required. For example, if the data is captured a special/restricted zone, prior approvals may be required.
    

## Data Lablling
Data labelling or Data Annotation is the process of tagging or marking the object(s) of
interest in an image or a video. Tagging of objects help the machine learning algorithm
to detect and recognise them and helps to train the model for accurate predictions

### What is Data Labelling
In machine learning, data labeling (Also termed as Data Annotation) is the
process of identifying raw data (images, text files, videos, etc.) and adding one or more
meaningful and informative labels to provide context so that a machine learning model
can learn from it. For example, labels might indicate whether a photo contains a bird or
car. Data labeling is required for a variety of use cases including computer vision, natural language processing, and speech recognition.

### What are the types of Data Labelling
Depending on the type of use case, variety of annotation types can be applied.
The most common annotation types are;

  * **Classification:** In image classification, the goal is to identify the object(s) within in an image.
  * **Object Detection:** Object detection goes a little more deeper than classification. In addition to the detection of the object, it identifies the location (bounding boxes) of the objects.
* **Image Segmentation:** Image segmentation goes still deeper by recognising and understanding each pixel in the image. Since this provides both what and what is notan object of interest, it is highly accurate but time consuming. Whereas, the othertwo types are faster with less accuracy.

### How does data labeling work?
In machine learning, a properly labeled dataset that is used as the objective
standard to train a model is often called “ground truth.” The accuracy of your trained
model will depend on the accuracy of the ground truth, so spending the time and effort
to ensure accurate data labeling is essential.

### Data Labeling in Image Processing
When building an image processing model, you first need to label images, pixels,
or create a border that fully encloses a digital image, known as a bounding box, to generate your training dataset. For example, you can classify images by quality type (like
product vs. lifestyle images) or content (what’s actually in the image itself), or you can
segment an image at the pixel level. You can then use this training data to build a model
that can be used to automatically categorise images, detect the location of objects, or
segment an image.

#### Best Practises in Data Labeling
There are many techniques to improve the efficiency and accuracy of data labeling. Some
of these techniques include:
  * Perform labelling on a small sample of data and review, so that any corrections in
labelling can be captured early. This will help in saving time.
  * Review the labelling based on the model output and update them as necessary. A
correction in labelling can sometimes improve the model accuracy considerably.

### Using VIA VGG
Get started with VIA with the following steps.
  1. Define Attributes - The URL provides the step by step process to create attributes. https://www.robots.ox.ac.uk/~vgg/software/via/docs/creating_annotations.html Note:- Name and Class ("type" in the user guide) are the basic attributes required for annotation. Additional attributes can be added as required in the project.
  2. Load images - Once the required attributes are created, load the images for annotation using either of the options "Add Files" or "Add URL"
  3. Region Selection - Next, select the region shape for marking the bounding box. Polygon is the option usually used for our projects.
  4. Create Annotations - Using mouse, create bounding box around the object in the selected image. Samples are provided for reference later in the document.
  5. Once all the annotations are completed, save the annotation using Annotation -> Export Annotations option in the required format. The exported annotations
can be reloaded using Annotation -> Import Annotations option according to the
format originally saved.
The annotations can be saved as a project as well using Project -> Save option.
The project can be loaded at later point using the Load option.
Note:- Ideally, it is better to export the annotations/ save project on a regular
interval.
Detailed VIA user guide can be found by following the URL. (http://www.robots.
ox.ac.uk/~vgg/software/via/docs/user_guide.html)

Complete the entire list of topics mentioned under Basic Usage to get a good understanding of this tool usage