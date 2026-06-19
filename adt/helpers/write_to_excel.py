import os
import pandas as pd

"""
dict1 --> filtered_names_len_high_dot_counts
dict2 --> filtered_names_len_low_distance_counts
dict3 --> objects_names_less_than_2_seconds_dict
dict4 --> objects_names_high_dot_values
dict5 --> objects_names_low_distance_values
dict6 --> predictions_dict
dict7 --> goals_dict
"""

def write_to_excel(dict1, dict2, dict3, dict4, dict5, dict6, dict7, dataset_name, parameters_folder_name, current_time_s):
    
    # Define paths
    project_path = "Documents/projectaria_sandbox/projectaria_tools/projects/AriaDigitalTwinDatasetTools/object_anticipation/adt/"
    excel_folder = os.path.join(project_path, 'utils', 'excel', dataset_name, parameters_folder_name)
    os.makedirs(excel_folder, exist_ok=True)
    
    # Defne file name
    excel_file = os.path.join(excel_folder, f"dictionaries_{dataset_name}_{parameters_folder_name}.xlsx")
    
    # Define a separator DataFrame to insert between appends (with a timestamp for clarity)
    separator_row = pd.DataFrame([[f'--- APPEND SEPARATOR {current_time_s} ---']], columns=['Name'])

    # Add a blank row to further separate the sections
    blank_row = pd.DataFrame([[""]], columns=['Name'])

    # Check if the file exists
    if os.path.exists(excel_file):
        # Load existing data
        df1_existing = pd.read_excel(excel_file, sheet_name='Dot_counts') if 'Dot_counts' in pd.ExcelFile(excel_file).sheet_names else pd.DataFrame()
        df2_existing = pd.read_excel(excel_file, sheet_name='Distance_counts') if 'Distance_counts' in pd.ExcelFile(excel_file).sheet_names else pd.DataFrame()
        df3_existing = pd.read_excel(excel_file, sheet_name='Time_less_2') if 'Time_less_2' in pd.ExcelFile(excel_file).sheet_names else pd.DataFrame()
        df4_existing = pd.read_excel(excel_file, sheet_name='Dot_values') if 'Dot_values' in pd.ExcelFile(excel_file).sheet_names else pd.DataFrame()
        df5_existing = pd.read_excel(excel_file, sheet_name='Distance_values') if 'Distance_values' in pd.ExcelFile(excel_file).sheet_names else pd.DataFrame()
        df6_existing = pd.read_excel(excel_file, sheet_name='Predicted') if 'Predicted' in pd.ExcelFile(excel_file).sheet_names else pd.DataFrame()
        df7_existing = pd.read_excel(excel_file, sheet_name='Goal') if 'Goal' in pd.ExcelFile(excel_file).sheet_names else pd.DataFrame()
        df_common_existing = pd.read_excel(excel_file, sheet_name='Common') if 'Common' in pd.ExcelFile(excel_file).sheet_names else pd.DataFrame()
        df_common_at_least_two_existing = pd.read_excel(excel_file, sheet_name='Common_at_least_2') if 'Common_at_least_2' in pd.ExcelFile(excel_file).sheet_names else pd.DataFrame()
        merged_df_existing = pd.read_excel(excel_file, sheet_name='Combined Lists') if 'Combined Lists' in pd.ExcelFile(excel_file).sheet_names else pd.DataFrame()
    else:
        # Initialize empty dataframes if no existing data
        df1_existing = df2_existing = df3_existing = df4_existing = df5_existing = pd.DataFrame()
        df6_existing = df7_existing = df_common_existing = df_common_at_least_two_existing = merged_df_existing = pd.DataFrame()

    # Convert the dictionaries to dataframes
    df1 = pd.DataFrame(list(dict1.items()), columns=['Name', 'Dot - Counts'])
    df2 = pd.DataFrame(list(dict2.items()), columns=['Name', 'Distance - Counts'])
    df3 = pd.DataFrame(list(dict3.items()), columns=['Name', 'Time less than 2'])
    df4 = pd.DataFrame(list(dict4.items()), columns=['Name', 'Dot - Value'])
    df5 = pd.DataFrame(list(dict5.items()), columns=['Name', 'Distance - Value'])
    df6 = pd.DataFrame(list(dict6.items()), columns=['Time', 'Predicted Objects'])
    df7 = pd.DataFrame(list(dict7.items()), columns=['Time', 'Goal'])
    
     # Merge the dataframes
    merged_df = pd.merge(df1, df2, on='Name', how='outer')
    merged_df = pd.merge(merged_df, df3, on='Name', how='outer')
    merged_df = pd.merge(merged_df, df4, on='Name', how='outer')
    merged_df = pd.merge(merged_df, df5, on='Name', how='outer')

    # Find common objects dot counts / distance counts / time_less_than_2
    common_names = set(df1['Name']).intersection(df2['Name']).intersection(df3['Name'])

    # Create a DataFrame for common objects
    df_common = pd.DataFrame({'Name': list(common_names)})
    df_common['Dot'] = df_common['Name'].map(dict1)
    df_common['Distance'] = df_common['Name'].map(dict2)
    df_common['Time'] = df_common['Name'].map(dict3)

    # Find objects common in at least two lists among the three (dot / distance / time less than 2)
    at_least_common_names = set(df1['Name']).intersection(df2['Name']).union(
                             set(df1['Name']).intersection(df3['Name'])).union(
                             set(df2['Name']).intersection(df3['Name']))

    # Create a DataFrame for common objects in at least two lists
    df_common_at_least_two = pd.DataFrame({'Name': list(at_least_common_names)})
    df_common_at_least_two['Dot'] = df_common_at_least_two['Name'].map(dict1)
    df_common_at_least_two['Distance'] = df_common_at_least_two['Name'].map(dict2)
    df_common_at_least_two['Time'] = df_common_at_least_two['Name'].map(dict3)
    
    # Insert a blank row and separator before appending new data
    def append_with_separator(existing_df, new_df):
            # Append blank row and separator
        combined_df = pd.concat([existing_df, blank_row, separator_row, new_df], ignore_index=True)
        return combined_df
    
    # Append the new data with separator
    df1_combined = append_with_separator(df1_existing, df1)
    df2_combined = append_with_separator(df2_existing, df2)
    df3_combined = append_with_separator(df3_existing, df3)
    df4_combined = append_with_separator(df4_existing, df4)
    df5_combined = append_with_separator(df5_existing, df5)
    df6_combined = append_with_separator(df6_existing, df6)
    df7_combined = append_with_separator(df7_existing, df7)
    df_common_combined = append_with_separator(df_common_existing, df_common)
    df_common_at_least_two_combined = append_with_separator(df_common_at_least_two_existing, df_common_at_least_two)
    merged_df_combined = append_with_separator(merged_df_existing, merged_df)

    # Save to Excel
    with pd.ExcelWriter(excel_file, engine='openpyxl', mode='w') as writer:
        df1_combined.to_excel(writer, sheet_name='Dot_counts', index=False)
        df2_combined.to_excel(writer, sheet_name='Distance_counts', index=False)
        df3_combined.to_excel(writer, sheet_name='Time_less_2', index=False)
        df4_combined.to_excel(writer, sheet_name='Dot_values', index=False)
        df5_combined.to_excel(writer, sheet_name='Distance_values', index=False)       
        df6_combined.to_excel(writer, sheet_name='Predicted', index=False)
        df7_combined.to_excel(writer, sheet_name='Goal', index=False)

        # Common Data
        df_common_combined.to_excel(writer, sheet_name='Common', index=False)
        df_common_at_least_two_combined.to_excel(writer, sheet_name='Common_at_least_2', index=False)
        merged_df_combined.to_excel(writer, sheet_name='Combined Lists', index=False)
        
# def write_to_excel(dict1, dict2, dict3, dict4, dict5, dict6, dict7, dataset_name, parameters_folder_name, current_time_s):
#     # Define paths
#     project_path = "Documents/projectaria_sandbox/projectaria_tools/projects/AriaDigitalTwinDatasetTools/object_anticipation/adt/"
#     excel_folder = os.path.join(project_path, 'utils', 'excel', dataset_name, parameters_folder_name)
#     os.makedirs(excel_folder, exist_ok=True)
    
#     excel_file = os.path.join(excel_folder, f"dictionaries_{dataset_name}_{parameters_folder_name}.xlsx")
    
#     # Define a separator DataFrame to insert between appends (with a timestamp for clarity)
#     separator_df = pd.DataFrame([[f'--- APPEND SEPARATOR {current_time_s} ---']])

#     # Add a blank row to further separate the sections
#     blank_row_df = pd.DataFrame([[""]])

#     # Initialize dataframes for existing data
#     if os.path.exists(excel_file):
#         with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
#             existing_sheets = writer.book.sheetnames
            
#             # Load the existing data
#             df1_existing = pd.read_excel(excel_file, sheet_name='Dot_counts') if 'Dot_counts' in existing_sheets else pd.DataFrame()
#             df2_existing = pd.read_excel(excel_file, sheet_name='Distance_counts') if 'Distance_counts' in existing_sheets else pd.DataFrame()
#             df3_existing = pd.read_excel(excel_file, sheet_name='Dot_values') if 'Dot_values' in existing_sheets else pd.DataFrame()
#             df4_existing = pd.read_excel(excel_file, sheet_name='Distance_values') if 'Distance_values' in existing_sheets else pd.DataFrame()
#             df5_existing = pd.read_excel(excel_file, sheet_name='Time_less_2') if 'Time_less_2' in existing_sheets else pd.DataFrame()
#             df6_existing = pd.read_excel(excel_file, sheet_name='Predicted') if 'Predicted' in existing_sheets else pd.DataFrame()
#             df7_existing = pd.read_excel(excel_file, sheet_name='Goal') if 'Goal' in existing_sheets else pd.DataFrame()
            
#             # Insert the blank row followed by the separator row before appending new data
#             blank_row_df.to_excel(writer, sheet_name='Dot_counts', index=False, header=False, startrow=len(df1_existing) + 1)
#             separator_df.to_excel(writer, sheet_name='Dot_counts', index=False, header=False, startrow=len(df1_existing) + 2)
            
#             blank_row_df.to_excel(writer, sheet_name='Distance_counts', index=False, header=False, startrow=len(df2_existing) + 1)
#             separator_df.to_excel(writer, sheet_name='Distance_counts', index=False, header=False, startrow=len(df2_existing) + 2)
            
#             blank_row_df.to_excel(writer, sheet_name='Dot_values', index=False, header=False, startrow=len(df3_existing) + 1)
#             separator_df.to_excel(writer, sheet_name='Dot_values', index=False, header=False, startrow=len(df3_existing) + 2)
            
#             blank_row_df.to_excel(writer, sheet_name='Distance_values', index=False, header=False, startrow=len(df4_existing) + 1)
#             separator_df.to_excel(writer, sheet_name='Distance_values', index=False, header=False, startrow=len(df4_existing) + 2)
            
#             blank_row_df.to_excel(writer, sheet_name='Time_less_2', index=False, header=False, startrow=len(df5_existing) + 1)
#             separator_df.to_excel(writer, sheet_name='Time_less_2', index=False, header=False, startrow=len(df5_existing) + 2)
            
#             blank_row_df.to_excel(writer, sheet_name='Predicted', index=False, header=False, startrow=len(df6_existing) + 1)
#             separator_df.to_excel(writer, sheet_name='Predicted', index=False, header=False, startrow=len(df6_existing) + 2)
            
#             blank_row_df.to_excel(writer, sheet_name='Goal', index=False, header=False, startrow=len(df7_existing) + 1)
#             separator_df.to_excel(writer, sheet_name='Goal', index=False, header=False, startrow=len(df7_existing) + 2)
    
#     else:
#         # If the file doesn't exist, initialize empty DataFrames
#         df1_existing = df2_existing = df3_existing = df4_existing = df5_existing = pd.DataFrame()
#         df6_existing = df7_existing = pd.DataFrame()

#     # Convert the dictionaries to dataframes
#     df1 = pd.DataFrame(list(dict1.items()), columns=['Name', 'Dot - Counts'])
#     df2 = pd.DataFrame(list(dict2.items()), columns=['Name', 'Distance - Counts'])
#     df3 = pd.DataFrame(list(dict3.items()), columns=['Name', 'Dot - Value'])
#     df4 = pd.DataFrame(list(dict4.items()), columns=['Name', 'Distance - Value'])
#     df5 = pd.DataFrame(list(dict5.items()), columns=['Name', 'Time less than 2'])
#     df6 = pd.DataFrame(list(dict6.items()), columns=['Time', 'Predicted Objects'])
#     df7 = pd.DataFrame(list(dict7.items()), columns=['Time', 'Goal'])
    
#     # Append the new data to the existing data
#     df1_combined = pd.concat([df1_existing, df1], ignore_index=True)
#     df2_combined = pd.concat([df2_existing, df2], ignore_index=True)
#     df3_combined = pd.concat([df3_existing, df3], ignore_index=True)
#     df4_combined = pd.concat([df4_existing, df4], ignore_index=True)
#     df5_combined = pd.concat([df5_existing, df5], ignore_index=True)
#     df6_combined = pd.concat([df6_existing, df6], ignore_index=True)
#     df7_combined = pd.concat([df7_existing, df7], ignore_index=True)

#     # Save to Excel
#     with pd.ExcelWriter(excel_file, engine='openpyxl', mode='w') as writer:
#         df1_combined.to_excel(writer, sheet_name='Dot_counts', index=False)
#         df2_combined.to_excel(writer, sheet_name='Distance_counts', index=False)
#         df3_combined.to_excel(writer, sheet_name='Dot_values', index=False)
#         df4_combined.to_excel(writer, sheet_name='Distance_values', index=False)
#         df5_combined.to_excel(writer, sheet_name='Time_less_2', index=False)
#         df6_combined.to_excel(writer, sheet_name='Predicted', index=False)
#         df7_combined.to_excel(writer, sheet_name='Goal', index=False)
