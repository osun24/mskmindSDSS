from PIL import Image
from PIL import ImageDraw, ImageFont

# Load the four images
image1 = Image.open('km_plot_SMOKING_STATUS.png')
image2 = Image.open('km_plot_IMMUNOTHERAPY.png')
image3 = Image.open('km_plot_EGFR_DRIVER.png')
image4 = Image.open('km_plot_STK11_DRIVER.png')

# Determine the size of each image (assuming all images are the same size)
width, height = image1.size

# Create a new blank image with dimensions to fit all four images in a 2x2 grid
new_image = Image.new('RGB', (width * 2, height * 2))


"""# Load a font
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
except IOError:
    font = ImageFont.load_default()

# Define labels and their positions
labels = ['A)', 'B)', 'C)', 'D)']
positions = [(20, 10), (width + 20, 10), (10, height + 10), (width + 10, height + 10)]
"""

new_image.paste(image1, (0, 0))
new_image.paste(image2, (width, 0))
new_image.paste(image3, (0, height))
new_image.paste(image4, (width, height))

# Save the resulting image
new_image.save('km/km_quad_SMOKE_IO_EGFR_STK11.png')