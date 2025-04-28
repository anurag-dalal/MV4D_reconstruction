import ptlflow
import cv2

from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils import flow_utils

# Get an initialized model from PTLFlow
model = ptlflow.get_model("dpflow", ckpt_path="things")
model.eval()

# Load the two images
img1 = cv2.imread("frame_0001.png")
img2 = cv2.imread("frame_0002.png")

# IOAdapter is a helper to transform the two images into the input format accepted by PTLFlow models
io_adapter = IOAdapter(model, img1.shape[:2])
inputs = io_adapter.prepare_inputs([img1, img2])
print('------', inputs["images"].shape, inputs["images"].dtype)  # Should be (1, 2, H, W, C) where H and W are the height and width of the images
# # Forward the inputs to obtain the model predictions
# predictions = model(inputs)

# # Visualize the predicted flow
# flow = predictions["flows"][0, 0]  # Remove batch and sequence dimensions
# flow = flow.permute(1, 2, 0)  # change from CHW to HWC shape
# flow = flow.detach().numpy()
# flow_viz = flow_utils.flow_to_rgb(flow)  # Represent the flow as RGB colors
# flow_viz = cv2.cvtColor(flow_viz, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR format
# # cv2.imshow("Optical Flow", flow_viz)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# cv2.imwrite("flow_viz.png", flow_viz)  # Save the flow visualization