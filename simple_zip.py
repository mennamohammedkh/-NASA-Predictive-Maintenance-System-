import os
import zipfile

def create_submission_zip():
    # المكان اللي انت فيه دلوقتي
    main_folder = r"D:\CS 3RD Year\2nd Semester\Intelligent_Ass_Projects\Predictive_Maintenance_NASA"
    
    zip_name = "Predictive_Maintenance_Submission.zip"
    zip_path = os.path.join(main_folder, zip_name)
    
    # لو في ZIP قديم، امسحه
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"🗑️ Removed old {zip_name}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        
        # 1. ضيف الـ Notebook
        notebook_path = os.path.join(main_folder, "1- EDA_and_Cleaning.ipynb")
        if os.path.exists(notebook_path):
            zipf.write(notebook_path, "1- EDA_and_Cleaning.ipynb")
            print("✅ Added: 1- EDA_and_Cleaning.ipynb")
        else:
            print("❌ Notebook not found! Looking for .ipynb file...")
            # لو مش موجود، دور على أي ملف .ipynb
            for f in os.listdir(main_folder):
                if f.endswith('.ipynb'):
                    zipf.write(os.path.join(main_folder, f), f)
                    print(f"✅ Added: {f}")
                    break
        
        # 2. ضيف requirements.txt
        req_path = os.path.join(main_folder, "requirements.txt")
        if os.path.exists(req_path):
            zipf.write(req_path, "requirements.txt")
            print("✅ Added: requirements.txt")
        else:
            print("⚠️ requirements.txt not found")
        
        # 3. ضيف فولدر models بالكامل
        models_path = os.path.join(main_folder, "models")
        if os.path.exists(models_path):
            for file in os.listdir(models_path):
                file_path = os.path.join(models_path, file)
                if os.path.isfile(file_path):
                    zipf.write(file_path, f"models/{file}")
                    print(f"✅ Added: models/{file}")
        else:
            print("⚠️ models folder not found")
        
        # 4. ضيف README.md لو موجود
        readme_path = os.path.join(main_folder, "README.md")
        if os.path.exists(readme_path) and os.path.getsize(readme_path) > 0:
            zipf.write(readme_path, "README.md")
            print("✅ Added: README.md")
    
    # اعرض النتيجة
    if os.path.exists(zip_path):
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"\n{'='*50}")
        print(f"📦 Created: {zip_name}")
        print(f"📏 Size: {size_mb:.2f} MB")
        print(f"📍 Location: {main_folder}")
        print(f"{'='*50}")
        
        if size_mb > 20:
            print(f"⚠️ WARNING: Size {size_mb:.2f} MB exceeds 20 MB limit!")
        else:
            print(f"✅ File size OK (under 20 MB)")
    else:
        print(f"\n❌ Failed to create zip file!")

if __name__ == "__main__":
    create_submission_zip()