%% 
% First Filtering for the right Data

file = {'000.txt', '002.txt', '003.txt', '005.txt', '006.txt', '007.txt', '008.txt', '009.txt', '010.txt', '011.txt', '013.txt', '401.txt', '402.txt', '403.txt', '404.txt', '405.txt', '406.txt', '407.txt', '408.txt', '409.txt', '410.txt', '411.txt', '412.txt', '413.txt', '414.txt', '415.txt', '416.txt', '418.txt', '419.txt', '420.txt', '421.txt', '422.txt', '423.txt', '424.txt', '425.txt', '426.txt', '427.txt'};  %, '4000.txt', '4001.txt', '4002.txt', '4003.txt', '4004.txt', '4005.txt', '4006.txt', '4007.txt', '4008.txt', '4009.txt', '4010.txt', '4011.txt', '4012.txt', '4013.txt', '4014.txt', '4015.txt', '4016.txt', '4017.txt', '4018.txt', '4019.txt', '4020.txt', '4021.txt', '4022.txt', '4023.txt', '4024.txt', '4025.txt', '4026.txt', '4027.txt', '4028.txt', '4029.txt', '4030.txt', '4031.txt', '4032.txt', '4033.txt', '4034.txt', '4035.txt', '4036.txt', '4037.txt', '4038.txt', '4039.txt', '4040.txt', '4041.txt', '4042.txt', '4043.txt', '4044.txt', '4045.txt', '4046.txt', '4047.txt', '4049.txt', '4050.txt', '4051.txt', '4052.txt', '4054.txt', '4055.txt', '4056.txt', '4057.txt', '4058.txt', '4059.txt', '4060.txt', '4061.txt', '4062.txt', '4064.txt', '4065.txt', '4066.txt', '4067.txt', '4068.txt', '4069.txt', '4070.txt', '4071.txt', '4072.txt', '4074.txt', '4075.txt', '4076.txt', '4077.txt', '4078.txt', '4079.txt', '4081.txt', '4082.txt', '4083.txt', '4084.txt', '4085.txt', '4086.txt', '4087.txt', '4088.txt', '4089.txt', '4090.txt', '4091.txt', '4092.txt', '4096.txt', '4097.txt', '4098.txt', '4099.txt', '4100.txt', '4101.txt', '4102.txt', '4106.txt', '4107.txt', '4108.txt', '4109.txt', '4110.txt', '4111.txt', '4112.txt', '4113.txt', '4114.txt', '4115.txt', '4116.txt', '4117.txt', '4118.txt', '4119.txt', '4120.txt'};
age = [53, 17, 46, 38, 32, 51, 39, 24, 55, 17, 20, 39, 12, 10, 13, 5, 15, 15, 6, 13, 10, 12, 8, 8, 12, 11, 11, 11, 12, 10, 14, 10, 12, 7, 11, 7, 12, 9, 0.42, 0.58, 1.08, 0.25, 0.17, 0.33, 0.25, 0.42, 0.75, 0.083, 0.5, 0.42, 0.42, 1.75, 0.75, 1.17, 1.17, 0.33, 0.42, 0.42, 2.25, 0.42, 0.75, 0.083, 1, 0.67, 0.5, 0.58, 0.5, 0.75, 0.58, 0.67, 0.33, 0.75, 0.42, 0.33, 0.083, 0.083, 1.08, 1.67, 1, 2.83, 2.75, 0.5, 0.83, 2.08, 0.17, 1.75, 0.83, 0.75, 0.42, 0.25, 0.25, 0.083, 0.33, 0.33, 0.083, 0.5, 0.67, 5.17, 0.083, 2.17, 0.33, 0.83, 0.33, 1.67, 3.42, 0.083, 1.5, 0.17, 0.25, 0.75, 0.58, 0.58, 0.58, 0.17, 0.33, 0.17, 0.33, 1.5, 0.17, 0.92, 0.5, 0.42, 0.42, 0.083, 0.42, 1.08, 4.92, 1.5, 2.5, 2.92, 3.00, 0.67, 2.08, 2.58, 3.42, 2.67, 5.83, 6.00, 4.00, 5.42, 5.00, 0.67, 4.5, 5.92, 1.42];

% Find indices where age is greater than 17
selected_indices = find(age > 17);
file_names = file(selected_indices);
n = length(file_names);
 
% disp(file_names);
% disp(n);

max_age = max(age(selected_indices));
min_age = min(age(selected_indices));

disp("n=" + n);
disp("Age Range=" + min_age + " - " + max_age);

%% 


msize = 1000;
file_name = 'Self_generated_RRs/RR_Matrix_Full.txt';


matrix = zeros(14, msize);

% Loop through each column and randomly select 14 values from random parts of the files
for i = 1:msize
    % Randomly select a file name
    selected_file = datasample(file_names, 1, 'Replace', false);
    
    % Read the data from the file
    file_path = ['physioNet_RR/' selected_file{1}];

    % Initialize a flag for non-numeric data
    non_numeric_flag = true;
    
    % Keep trying until we read a numeric file successfully
    while non_numeric_flag
        try
            % Attempt to read the data from the file
            data = dlmread(file_path);
            
            % If successful, set the flag to false
            non_numeric_flag = false;
        catch
            % If an error occurs, try again with a different file
            selected_file = datasample(file_names, 1, 'Replace', false);
            file_path = ['physioNet_RR/' selected_file{1}];
         end
    end
    
    % Randomly select a starting index to extract 14 values
    start_index = randi(length(data) - 13);
    
    % Extract the 14 values
    selected_values = data(start_index:start_index+13);
    
    % Assign the selected values to the matrix column
    matrix(:, i) = selected_values(:);
    disp((i/msize)*100+ "%");
end

% Display the matrix
 
%% PLOT RR INTERVALS 

% Number of columns to plot (adjust as needed)
select = msize;

% Plot the first 'select' columns of the matrix
figure;
plot(matrix(:, 1:select), 'LineWidth', 1);
xlabel('Time Steps');
ylabel('RR Intervals');
title(['RR Intervals Over Time | n=' num2str(n) ', Age Range=' num2str(min_age) '-' num2str(max_age) ', Selected Columns=' num2str(select)]);
 
 
%% OUTLIERS
 

 

% Set the threshold for outliers (adjust as needed)
threshold = 2.5;

% Initialize a variable to store indices of columns with outliers
columns_with_outliers_indices = [];

% Loop through each column of the matrix
for col = 1:size(matrix, 2)
    % Calculate the mean and standard deviation for the column
    mean_col = mean(matrix(:, col));
    std_col = std(matrix(:, col));
    
    % Check for outliers in the column
    outliers = abs(matrix(:, col) - mean_col) > threshold * std_col;
    
    % If any outliers are found, store the column index
    if any(outliers)
        columns_with_outliers_indices = [columns_with_outliers_indices, col];
    end
end

% Plot the columns with outliers using a line plot
figure;
plot(matrix(:, columns_with_outliers_indices));

% Customize the plot
xlabel('Row Index');
ylabel('Values');
missedHB = (length(columns_with_outliers_indices)/length(matrix))*100;
title(sprintf('Columns with Outliers: %d %%',missedHB));

% Add a legend (you can customize the legend entry based on your data)
%legend(cellstr(num2str(columns_with_outliers_indices')));

% Add grid lines
grid on;




%% 
% Assuming you have a matrix called data_matrix of size 14x1000

data_matrix = matrix;
matrix_size = size(data_matrix);
num_rows = matrix_size(1);
num_columns = matrix_size(2);

% Specify the percentage of columns to modify (5% in this case)
percentage_to_modify = 5;

% Calculate the number of columns to modify
num_columns_to_modify = round(percentage_to_modify / 100 * num_columns);

% Get the list of columns that are not in columns_with_outliers_indices
available_columns = setdiff(1:num_columns, columns_with_outliers_indices);

% Randomly select columns to modify from the available_columns
columns_to_modify = datasample(available_columns, num_columns_to_modify, 'Replace', false);

% Randomly select a row to modify in each selected column
rows_to_modify = randi(num_rows, 1, num_columns_to_modify);

 
% Double a single value in the selected rows and columns
for i = 1:num_columns_to_modify
    row_index = rows_to_modify(i);
    col_index = columns_to_modify(i);
    data_matrix(row_index, col_index) = 2 * data_matrix(row_index, col_index);
end

%% 

% Plot the modified columns
figure;
plot(data_matrix(:, columns_to_modify));
title('Added 5% of Missed Heart Beats to the Data');
xlabel('Row Index');
ylabel('Value');

%% PLOT RR All 

% Number of columns to plot (adjust as needed)
select = 44;

% Plot the first 'select' columns of the matrix
figure;
plot(data_matrix(:, 1:select), 'LineWidth', 1);
xlabel('Time Steps');
ylabel('RR Intervals');
title(['RR Intervals Over Time | n=' num2str(n) ', Age Range=' num2str(min_age) '-' num2str(max_age) ', Selected Columns=' num2str(select)]);
 
 


%% PLOT RR INTERVALS  FINAL

 select = msize;

% Plot the first 'select' columns of the matrix
figure;
plot(data_matrix);
xlabel('Time Steps');
ylabel('RR Intervals');
title(['RR Intervals Over Time | n=' num2str(n) ', Age Range=' num2str(min_age) '-' num2str(max_age) ', Selected Columns=' num2str(select)]);
 


%% Save that shit



T_data_matrix = data_matrix';
% Assuming T_data_matrix is your MATLAB matrix
[unique_rows, ~, idx] = unique(T_data_matrix, 'rows');
duplicate_indices = setdiff(1:size(T_data_matrix, 1), idx);

% Check if there are any duplicates
if ~isempty(duplicate_indices)
    fprintf('Found %d duplicate rows.\n __ Therfore Dataset not not Saved', length(duplicate_indices));
    % Optionally, print the indices of the duplicate rows
    disp(duplicate_indices);
else
    fprintf('No duplicate rows found.\n');
    dlmwrite(file_name, T_data_matrix, 'delimiter', '\t', 'precision', 6);
    fprintf('RR_Dataset SAVED----.\n');

end



 


