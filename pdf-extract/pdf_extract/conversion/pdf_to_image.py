import os
import glob
import pdf2image as p2i

def pdf_to_image(pdf_directory: str, images_directory: str, img_extension: str = "jpg", **kwargs):
    
    # let us verify if the extension of the images are valid
    if img_extension == 'jpg':
        
        format = "JPEG"
    
    elif img_extension == 'png':
        
        format = "PNG"
    
    else:
        
        raise ValueError("You can choose only between .jpg or .png extensions")
    
    # let us verify if the directories exist
    if not os.path.exists(pdf_directory):
        
        raise OSError("You must provide a existing pdf directory !")
    
    if not os.path.exists(images_directory):
        
        raise OSError("You must provide an existing images' directory !")
    
    # let us recuperate the pages of each pdf in the directory and 
    # place the their converted versions in a new directory with the same name
    # as that of the pdf file
    pdfs = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    
    for pdf in pdfs:
        
        # recuperate the name of the pdf file
        name = os.path.splitext(os.path.basename(pdf))[0]
        
        # extract the pages
        pages = p2i.convert_from_path(pdf, **kwargs)
        
        # let us create a new directory for the current pdf file
        new_current_path = os.path.join(images_directory, name)
        
        if not os.path.exists(new_current_path):
            
            os.makedirs(new_current_path)
            
        else:
            
            raise OSError(f"The directory {new_current_path} already exist and cannot be overwrited !")

        # let us save the image in new directory
        for i, page in enumerate(pages):
            
            page.save(os.path.join(new_current_path, f"{i}.{img_extension}"), format=format)
        
        print(f"The found pages in the pdf file {name} were successfully extracted !")
    
    
    
