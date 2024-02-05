% Define the path to the bag file and the output folder
bagPath = 'C:\Users\axyzh\Documents\autodrive\bags_matlab\run_5.bag';
outputFolder = 'C:\Users\axyzh\Documents\autodrive\bags_matlab\bag_images';

% Load the ROS bag
bag = rosbag(bagPath);

% Select the topic with image data
topic = '/zedx/zed_node/rgb/image_rect_color';
imageBag = select(bag, 'Topic', topic);

% Read messages
msgs = readMessages(imageBag, 'DataFormat', 'struct');

% Check if the output folder exists, if not, create it
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Loop through messages and save frames to the specified folder
for i = 1:length(msgs)
    frame = rosReadImage(msgs{i});
    filename = fullfile(outputFolder, sprintf('frame_%d.png', i));
    imwrite(frame, filename);
end