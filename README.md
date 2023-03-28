# LGL-VAE
## Latent Generative Landscape - Variational Autoencoder
### Ziegler, C., Martin, J., Sinner, C. and Morcos, F. (TBD) "Latent Generative Landscapes as Maps of Functional Diversity in Protein Sequence Space." TBD
***
## **Instructions for Use**
- **Installation: Docker (Installation time less than 30 min if Docker already installed)**
    - [Download and install Docker here](https://docs.docker.com/get-docker/)
    - Clone/download this git repository to your system.
    - Navigate command line to unzipped directory and build the Docker image with this command:
        - ```docker build -f docker/Dockerfile -t lgl_vae .```
    - Then, run the image with this command:
        - ```docker run -p 5006:5006 -v <directory to save files to>:/docker lgl_vae```
    - Then, open http://localhost:5006 in your browser.
- **Using the interface to train a model**
    - Follow on screen instructions at the top of the initial web page to create an output folder, select an aligned fasta file, and train the model.
        - Training progress/LGL generation can be observed in the terminal.
    - When training has completed, load the model by selecting the file ```latent_generative_landscape.pkl``` in your model folder using the labeled File Select button, then hit "Plot LGL".
- **Visualizing sequences in the interface**
    - After training and loading a model, the "Plotting Tools" tab can be used to plot additional sequences and load sequence labels to quickly color the sequences by user defined labels.
    - Hit plot training data to plot the sequences used during training in the latent landscape. 
    - Under the Additional Sequences label, the "Plot Additional MSA as Gradient" button can be toggled to change the coloring of any additional sequences.
        - Plotting sequences as a gradient can be useful if you are plotting an evolutionary trajectory, with the fasta file containing the initial sequence at the top and the final sequence at the bottom (or vice versa).
        - Click the button to change the coloring to a solid color, useful for visualizing unordered groups of sequences.
    - **Additional sequences can be loaded** from a fasta file by selecting the browse button under the Additional Sequences label.
    - You can quickly label your sequences by functional attributes using the **CSV loading function**. 
        - Use the dropdown box to select which set of sequences you are labeling (Training or additional sequences).
        - Load the CSV for the selected sequence. CSV format is as follows:
            - First row is the Column label.
            - Subsequent rows are sequence labels. **You must have the same number of rows as sequences**, and they are applied to your sequences sequentially (first sequence in training MSA will correspond to the first non-column label row in your CSV).
        - Choose the CSV column, and the plot will update colors based on the labels you provided in that column.
    - **Using the lasso tool**, you can select encoded sequence points and their fasta header/sequences will be displayed in the lower right text box.
    - **The Legend Editing Tab** can be used to selectively remove sets of sequences that have a specific label from the main plot.
***
## **Troubleshooting**
- **I can't find the .pkl file to load the model!**
    - When setting the volume in Docker with ```-v <directory to save files to>:/docker```, try and use the full path to the LGL-VAE install directory.
- **Training the model and generating the landscape takes a very long time!**
    - Very long sequences or sequence sets with a large number of sequences in them will take substantially longer to process. To generate an LGL on more powerful hardware that cannot run a bokeh server, see additional scripts provided for training the model on another system without using the interface.
- **I get a Tensorflow error while training and it does not finish!**
    - Ensure all of the sequences in your training fasta are the same length.
- **Additional sequences I plot fail to show up and I get a crazy Tensorflow error!**
    -  All sequences must be the same length as the sequences you used to train the model with.
- **I picked a CSV column, but nothing happend and it gave me a pandas error like "N cannot be fit into M"!**
    - Whichever sequence file you select, the CSV you load must have the same number of rows as there are sequences + the additional row for the Columnn Headers.
- **I'm on an M1 Mac and the docker container is giving me errors like unsupported architecture or "can't find the right Tensorflow version"!**
    - The Dockerfile must be modified for this to work on ARM architectures, we are currently working on it. 
 
All other issues, please feel free to open a support ticket and we will help you out!
    
***
## **Demo (~30 min)**
- **Use fasta file to train network**
- **Use csv to relabel training sequences**
- **Can test additional sequences tool using original training fasta and csv**

***
## **Reproduction**
- **Use training and plotting files from Dryad (doi:10.5061/dryad.51c59zwbn)**
