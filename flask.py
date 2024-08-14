from flask import Flask, request, jsonify, send_file
import torch
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)

# Initialize your model here


def process_image(image_data):
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    # Convert image to tensor if necessary
    return image


@app.route('/run_algorithm', methods=['POST'])
def run_algorithm():
    try:
        # Get data from the request
        data = request.json
        image_data = data['image']  # Base64 encoded image
        label_or_target = data['label_or_target']
        targeted = data.get('targeted', False)
        epsilon = data.get('epsilon', 0.05)
        step = data.get('step', 100)
        lr = data.get('lr', 0.01)
        n = data.get('n', 10)
        sigma = data.get('sigma', 1e-3)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Process the image
        x_in = process_image(image_data)

        # Convert label_or_target to tensor
        label_tensor = torch.tensor(label_or_target).to(nes_instance.device)

        # Run the NES algorithm
        result_tensor = nes_instance(x_in, label_tensor, targeted, epsilon, step, lr, n, sigma)

        # Convert result tensor back to an image
        result_image = Image.fromarray((result_tensor.squeeze().cpu().numpy() * 255).astype('uint8'))

        # Save image to a bytes buffer
        buffered = BytesIO()
        result_image.save(buffered, format="PNG")
        encoded_result = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Return the result
        return jsonify({
            'status': 'success',
            'image': encoded_result,
            'additional_info': {
                # Add any additional info you need
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
