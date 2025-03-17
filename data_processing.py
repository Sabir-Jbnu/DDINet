import untangle
import pandas as pd
import os
from tqdm import tqdm
import pubchempy as pcp

# ============================ File Paths ============================
filename = "full_database.xml"
saveFile_seen = "seen_drs.csv"
saveFile_unseen = "unseen_drs.csv"
all_drugs_csv_path = "all_drug.csv"

#%% ============================ Parse XML and Extract Drug Data ============================
obj = untangle.parse(filename)
seen_drugs_list = []  # Approved drugs
unseen_drugs_list = []  # Experimental or investigational drugs

for drug in obj.drugbank.drug:
    drug_type = drug['type'].lower()

    if drug_type == "small molecule":
        drug_info = {
            "drugbank_id": None,
            "name": drug.name.cdata if hasattr(drug, 'name') else '',
            "cas": drug.cas_number.cdata if hasattr(drug, 'cas_number') else '',
            "smiles": None,
            "groups": []
        }

        if hasattr(drug, 'drugbank_id'):
            for id in drug.drugbank_id:
                if hasattr(id, 'primary') and id['primary'] == "true":
                    drug_info["drugbank_id"] = id.cdata
                    break

        if hasattr(drug, 'calculated_properties') and hasattr(drug.calculated_properties, 'property'):
            for prop in drug.calculated_properties.property:
                if prop.kind.cdata == "SMILES":
                    drug_info["smiles"] = prop.value.cdata

        if hasattr(drug, 'groups'):
            for group in drug.groups.group:
                drug_info["groups"].append(group.cdata if hasattr(group, 'cdata') else group)

        groups = set(drug_info["groups"])
        if 'approved' in groups:
            seen_drugs_list.append(drug_info)
        elif 'experimental' in groups or 'investigational' in groups:
            unseen_drugs_list.append(drug_info)

seen_drugs_df = pd.DataFrame(seen_drugs_list)
unseen_drugs_df = pd.DataFrame(unseen_drugs_list)

seen_drugs_df.to_csv(saveFile_seen, index=False)
unseen_drugs_df.to_csv(saveFile_unseen, index=False)

print(f"Files saved: Seen drugs (approved) to {saveFile_seen}, Unseen drugs (experimental/investigational) to {saveFile_unseen}")

#%% ============================ Filter and Normalize Data ============================
# Filter seen drugs (approved)
df_seen = pd.read_csv(saveFile_seen)
df_seen['groups'] = df_seen['groups'].str.strip()
filtered_df_seen = df_seen[df_seen['groups'] == "['approved']"]
filtered_df_seen.to_csv("filtered_approved_drugs.csv", index=False)
print("Filtered approved data has been saved. Rows:", len(filtered_df_seen))

# Filter unseen drugs (experimental/investigational)
df_unseen = pd.read_csv(saveFile_unseen)
df_unseen['groups'] = df_unseen['groups'].str.strip()
filtered_df_unseen = df_unseen[(df_unseen['groups'] == "['experimental']") | (df_unseen['groups'] == "['investigational']")]
filtered_df_unseen.to_csv("filtered_experimental_drugs.csv", index=False)
print("Filtered experimental/investigational data has been saved. Rows:", len(filtered_df_unseen))
#%% ============================ Load and Process DataFrames ============================
merged = pd.read_csv("DDI_data.csv")
seen = pd.read_csv("final_seen_drugs.csv")
unseen = pd.read_csv("final_unseen_drugs.csv")

merged['AB'] = merged['drug_A'] + merged['drug_B']
A = merged['drug_A'].tolist()
B = merged['drug_B'].tolist()
labels = merged['DDI'].tolist()

seen_list = seen.iloc[:, 1].tolist()
unseen_list = unseen.iloc[:, 1].tolist()

seen_set = set(seen_list)
unseen_set = set(unseen_list)

S1_train = []
S2_test = []
S3_test = []

for i in tqdm(range(len(A))):
    if (A[i] in seen_set and B[i] in unseen_set) or (A[i] in unseen_set and B[i] in seen_set):
        S2_test.append([A[i], B[i]])
    elif A[i] in unseen_set and B[i] in unseen_set:
        S3_test.append([A[i], B[i]])
    else:
        S1_train.append([A[i], B[i]])

print("S2_test length:", len(S2_test))
print("S3_test length:", len(S3_test))
print("S1_train length:", len(S1_train))

df_S1 = pd.DataFrame(S1_train, columns=['A', 'B'])
df_S2 = pd.DataFrame(S2_test, columns=['A', 'B'])
df_S3 = pd.DataFrame(S3_test, columns=['A', 'B'])

df_S1.to_csv("S1.csv", index=False) # train data
df_S2.to_csv("S2.csv", index=False) # seen-unseen test data
df_S3.to_csv("S3.csv", index=False) # unseen-unseen test data
#%%
