import numpy as np

# Declare a dictionary and arrays to store the data. #
data = {}
all_redshifts, all_regions, all_group_numbers, all_subgroup_numbers, all_disc_fractions = [], [], [], [], []

# Define redshift range and regions. #
redshifts = [8, 7, 6, 5]
tags = ['007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000']
regions = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '17', '18',
           '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36',
           '37', '38', '39']

# Loop over all regions for each redshift. #
for tag, redshift in zip(tags, redshifts):
    for region in regions:
        # Path the data is stored. #
        data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/JWST/' + region + '/' + tag + '/'

        # Try to load the 'group_numbers' to make sure that a specific region at a given redshift exists. If file
        # doesn't exist, continue to the next region. #
        try:
            group_numbers = np.load(data_path + 'group_numbers.npy')
        except FileNotFoundError:
            continue

        # Check if there are galaxies in a specific region at a given redshift. If yes, load their 'subgroup_numbers'
        # and 'disc_fractions' #
        if len(group_numbers) > 0.0:
            subgroup_numbers = np.load(data_path + 'subgroup_numbers.npy')
            disc_fractions = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        else:
            continue

        # Add the data from different regions together into one list. #
        all_group_numbers.extend(group_numbers)
        all_disc_fractions.extend(disc_fractions)
        all_subgroup_numbers.extend(subgroup_numbers)
        for i in range(len(group_numbers)):
            all_regions.append(region)
            all_redshifts.append(redshift)

# Create and save a dictionary with the data from all redshifts and regions. #
data['regions'] = all_regions
data['redshifts'] = all_redshifts
data['group_numbers'] = all_group_numbers
data['disc_fractions'] = all_disc_fractions
data['subgroup_numbers'] = all_subgroup_numbers
np.save('/cosma7/data/dp004/dc-irod1/FLARES/morph_data_JWST', data)

# Load the dictionary. #
morp_data = np.load('/cosma7/data/dp004/dc-irod1/FLARES/morph_data_JWST.npy', allow_pickle=True)
morp_data = morp_data.item()

print(morp_data['regions'])
print(morp_data['redshifts'])
print(morp_data['group_numbers'])
print(morp_data['disc_fractions'])
print(morp_data['subgroup_numbers'])
