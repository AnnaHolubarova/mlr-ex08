1. It runs on your phone: the Pages URL opens on your phone and classifies the live camera. → A screenshot from your phone with a leaf (or a photo of a leaf on another screen) and the probabilities.

Yes, it does run on my phone. The screenshot of the page is IMG_1370.jpg in this folder.

2. The same numbers as the notebook: → Upload check_image.jpg on the page and write its probabilities next to check_expected.json. Do they agree to three decimals?

From the page:
"Self-check against check_expected.json
Largest difference 0.000000 → agrees to three decimals; top class matches."


class	            notebook
                check_expected.json	        this page	        difference
apple	            0.006720	            0.006720	         -0.000000
cherry	            0.992578	            0.992578	         +0.000000
chestnut	        0.000554	            0.000554	         -0.000000
maple	            0.000148	            0.000148	         -0.000000


3. The preprocessing matches your model: → Find in index.html where the frame is cropped, resized to 96 × 96, divided by 255 and standardized with the mean and std, and in which order the channels go into the tensor. Does it match model_info.json and the transform of the notebook?

The preprocessing steps match the parameters in model_info.json. The image is cropped and resized (into NxN grid where N is from info.input_size -> 96), scaled (divided by 255 to cale them o a 0 to 1 range) and standardized and pixels are written into a float32 tensor.

4. Nothing leaves the phone: → Is the image sent anywhere? Which URLs does the page load, and why?
Images are not sent anywhere.

No image is ever uploaded, everything takes place inside a <canvas> element and converted into a float32array. The page loads two types of URLs - its own GitHub files to load the metadata, model weights etc. and ONNX runtime web (web assembly library) - this is necessary to execute the model locally.