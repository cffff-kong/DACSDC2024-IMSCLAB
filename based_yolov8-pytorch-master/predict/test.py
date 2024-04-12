# Read the contents of image_pred_m.txt
with open('image_pred_m.txt', 'r') as file_m:
    content_m = file_m.readlines()

# Read the contents of image_pred.txt
with open('image_pred.txt', 'r') as file:
    content = file.readlines()

# Remove the newline character from each line
content_m = [x.strip() for x in content_m]
content = [x.strip() for x in content]

# Calculate the difference between corresponding positions
diff_content = [str(float(x) - float(y)) for x, y in zip(content_m, content)]

# Save the result in a new text file
with open('diff.txt', 'w') as file_diff:
    file_diff.writelines(diff_content)