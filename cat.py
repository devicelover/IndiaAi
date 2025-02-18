import os
import pandas as pd
import argparse

# 1. Define your category mapping
CATEGORIES = {
    "Crime Against Women & Children": [
        "Rape Gang Rape",
        "Sexual Harassment",
        "Cyber Voyeurism",
        "Cyber Stalking",
        "Cyber Bullying",
        "Child Pornography Child Sexual Abuse Material (CSAM)",
        "Child Sexual Exploitative Material (CSEM)",
        "Publishing and transmitting obscene material sexually explicit material",
        "Computer Generated CSAM CSEM",
        "Fake Social Media Profile",
        "Defamation",
        "Cyber Blackmailing & Threatening",
        "Online Human Trafficking",
        "Others"
    ],
    "Financial Crimes": [
        "Investment Scam Trading Scam",
        "Online Job Fraud",
        "Tech Support Scam Customer Care Scam",
        "Online Loan Fraud",
        "Matrimonial Romance Scam Honey Trapping Scam",
        "Impersonation of Govt. Servant",
        "Cheating by Impersonation (other than Government Servant)",
        "SIM Swap Fraud",
        "Sextortion Nude Video Call",
        "Aadhar Enabled Payment System (AEPS) fraud - Biometric Cloning",
        "Identity Theft",
        "Courier Parcel Scam",
        "Phishing",
        "Online Shopping E-commerce Frauds",
        "Advance Fee",
        "Real Estate Rental Payment",
        "Others"
    ],
    "Cyber Attack Dependent Crimes": [
        "Malware Attack",
        "Ransomware Attack",
        "Hacking Defacement",
        "Data Breach Theft",
        "Tampering with computer source documents",
        "Denial of Service (DoS) Distributed Denial of Service (DDOS) attacks",
        "SQL Injection",
        "Cyber Terrorism",
        "Business Email Compromise Email Takeover",
        "Fake Social Media Profile",
        "Fake News",
        "Online Gambling Betting Frauds",
        "Provocative Speech for unlawful acts",
        "Social Media Account Hacking",
        "Cyber Pornography",
        "Cyber Blackmailing & Threatening",
        "Cyber Stalking Bullying",
        "Defamation",
        "Sending obscene material",
        "Intellectual Property (IPR) Thefts",
        "Cyber Enabled Human Trafficking Cyber Slavery",
        "Online Piracy",
        "Spoofing",
        "Others"
    ],
    # Add additional main categories & subcategories if needed
}


def process_files(input_folder, output_file):
    all_data = []
    found_subcats = {}  # Track which subcategories we have data for under each main category

    # 2. Walk through all directories inside input_folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # Only process Excel files
            if file.lower().endswith(".xlsx"):
                file_path = os.path.join(root, file)
                # Extract the main category from the folder structure
                relative_path = os.path.relpath(root, input_folder)

                # If the file is directly under input_folder, set main category as "Unknown"
                main_category = relative_path.split(os.sep)[0] if relative_path != '.' else "Unknown"

                # 3. Use the file name (without extension) as the subcategory
                subcategory = os.path.splitext(file)[0]

                print(f"Processing file: {file_path} (Main Category: {main_category}, Subcategory: {subcategory})")

                # Attempt to read Excel file
                try:
                    df = pd.read_excel(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

                # 4. Add columns
                df["Main Category"] = main_category
                df["Subcategory"] = subcategory
                all_data.append(df)

                # Keep track of found subcategories
                if main_category not in found_subcats:
                    found_subcats[main_category] = set()
                found_subcats[main_category].add(subcategory)

    # 5. Create placeholder rows for missing subcategories
    for main_cat, subcats in CATEGORIES.items():
        # If we never found data for this main_cat, initialize it
        if main_cat not in found_subcats:
            found_subcats[main_cat] = set()

        # For each subcategory in the mapping
        for subcat in subcats:
            # If the subcat was not found in the actual Excel files
            if subcat not in found_subcats[main_cat]:
                print(f"Missing data for {main_cat} -> {subcat}; adding placeholder row.")
                placeholder_df = pd.DataFrame({
                    "Main Category": [main_cat],
                    "Subcategory": [subcat]
                    # Add more columns here if you want placeholders for them
                })
                all_data.append(placeholder_df)

    # 6. Merge everything and save
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        merged_df.to_excel(output_file, index=False)
        print(f"Merged file saved to {output_file}")
    else:
        print("No Excel files found.")


def main():
    parser = argparse.ArgumentParser(
        description="Merge Excel files from multiple main categories/subcategories and add placeholder rows for missing subcategories."
    )
    parser.add_argument("input_folder", help="Root folder containing the Excel files organized by main category")
    parser.add_argument("output_file", help="Path to save the merged Excel file")
    args = parser.parse_args()

    process_files(args.input_folder, args.output_file)

if __name__ == "__main__":
    main()
