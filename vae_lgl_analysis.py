import base64
import bz2
import datetime
import io
import os
import pickle
from functools import partial
from itertools import cycle
from skimage.draw import polygon

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from Bio import SeqIO
from bokeh.layouts import column, row
from bokeh.models import (
    Select,
    Button,
    CheckboxGroup,
    ColorBar,
    ColumnDataSource,
    Div,
    Dropdown,
    FileInput,
    HoverTool,
    LassoSelectTool,
    LinearColorMapper,
    Panel,
    PreText,
    TextInput,
    Title,
    PointDrawTool,
    Legend,
    LegendItem,
    CDSView,
    GroupFilter,
    Div
)
from bokeh.events import SelectionGeometry
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import Tabs
from bokeh.palettes import Colorblind8, Set3, Viridis256, linear_palette
from bokeh.plotting import figure
from bokeh.server.server import Server
from dca.dca_class import dca
from keras.models import load_model
import plotly.graph_objects as go
import plotly.io as pio
import argparse


from model.generator import (
    get_fasta_file_dimensions,
    read_fasta_as_one_hot_encoded,
    seq_code,
    return_sequence
)
from model.model import VAE


def parse_args():
    parser = argparse.ArgumentParser(description='VAE LGL Analysis')
    parser.add_argument('--save-as-svg', action='store_true', 
                      help='Set output backend to SVG (default is PNG)')
    return parser.parse_args()


def vae_lgl_analysis_app(doc):
    model_params = {
        "batch_size": 16,
        "epochs": 1000,
        "seq_len": 0,
        "num_seqs": 0,
        "hu_num": 0,
        "activation": "relu",
        "l2_reg": 1e-4,
        "patience": 4,
        "train_start_time": 0,
    }  # you don't need to change these

    class mutable_variables:
        def __init__(self):
            self.fasta_name = ""
            self.num_seqs = 0
            self.seq_len = 0
            self.train_fasta = None
            self.train_model_name = ""
            self.the_model_name = ""
            self.the_model = None
            self.df = None
            self.base_cds = ColumnDataSource()
            self.grid_hamiltonian_plot = ""
            self.grid_hamiltonian_df = ""
            self.base_plot = ""
            self.label_df = None
            self.ldf_labels = list()
            self.legend_labels = list()
            self.grid_ranges = []
            self.pixels = 0
            self.plot_is_gradient = True
            self.landscape_seq_df = ""
            self.landscape_grid = None
            self.grid_seq = ''
            self.selected_postions = []
            self.bp_color_size = ["steelblue", 6]
            self.recolor_label = None
            self.csv_columns = []
            self.plotting_df = None
            self.glyphs = []

        def update_model(self, model_name):
            decoded = base64.b64decode(
                model_name
            )  # bokeh reads inputs in base64 so we need to decode
            bytes_decoded = io.BytesIO(decoded)  # this line decodes into bytes
            decompressed_pkl = bz2.BZ2File(bytes_decoded)
            (
                model_title,
                landscape,
                model_seq_len,
                vae_weights,
                training_seqs,
                landscape_seqs
            ) = pickle.load(decompressed_pkl)
            vae = VAE(
                num_aa_types=23,
                dim_latent_vars=2,
                dim_msa_vars=model_seq_len,
                num_hidden_units=model_seq_len * 3,
                activation_func=model_params["activation"],
                regularization=model_params["l2_reg"],
            )
            vae.set_weights(vae_weights)
            self.the_model = vae
            self.the_model_name = model_title
            self.landscape_grid = landscape

            decoded = base64.b64decode(
                training_seqs
            )  # bokeh reads inputs in base64 so we need to decode
            bytes_decoded = io.BytesIO(decoded).read()  # this line decodes into bytes
            text_obj = bytes_decoded.decode("UTF-8")  # now we convert into an text_obj
            lm.train_fasta = io.StringIO(text_obj)

            decoded = base64.b64decode(
                landscape_seqs
            )  # bokeh reads inputs in base64 so we need to decode
            bytes_decoded = io.BytesIO(decoded).read()  # this line decodes into bytes
            text_obj = bytes_decoded.decode("UTF-8")  # now we convert into an text_obj
            text_obj = io.StringIO(text_obj)
            lm.grid_seq = [record for record in SeqIO.parse(text_obj,"fasta")]

        def update_model_folders(self):
            self.model_folders = [
                folder for folder in os.listdir() if os.path.isdir(folder)
            ]

        def update_df(self, pandas_df):
            if self.df is None:
                self.df = pandas_df
            else:
                self.df = pd.concat([self.df, pandas_df])
                self.df.reset_index(drop=True)

        def update_base(self, plotscatter):
            self.base_plot = plotscatter

        def update_grid_df(self, dataframe):
            self.grid_hamiltonian_df = dataframe

        def update_grid_plot(self, plot_return):
            self.grid_hamiltonian_plot = plot_return

        def update_labeldf(self, labeldf):
            self.label_df = labeldf

        def update_legend_labels(self, newlabel):
            self.legend_labels.append(str(newlabel))
        
        def set_legend(self):
            legend_items = [LegendItem(label=self.legend_labels, renderers=self.glyphs)]
            legend = Legend(items=legend_items)
            return legend

        def update_ldf_labels(self, label_df):
            self.ldf_labels = list(label_df.columns)

        def init_basecds(self, cds):
            self.base_cds = cds

        def update_cds_column(self, labeldf_column, new_colors):
            refactor = np.array(self.df)
            idxs = np.where(refactor == lm.recolor_label)
            refactor[idxs[0],5] = lm.label_df[labeldf_column]
            refactor[idxs[0], 4] = new_colors
            lm.df = pd.DataFrame(data=refactor, columns=lm.df.columns).astype('string')

        def update_colors(self, colorlist):
            self.base_cds.data["colors"] = colorlist

        def update_labels(self, labellist):
            self.legend_labels.remove(self.recolor_label)
            self.legend_labels = self.legend_labels + labellist

        def remove_from_selection(self, unselected_label_list):
            for item in unselected_label_list:
                glyph = lm.glyphs[lm.legend_labels.index(item)]
                glyph.visible = False

                for legend_item in p.legend.items:
                    if legend_item.label['value'] == item:
                        p.legend.items.remove(legend_item)

        def add_from_selection(self, selected_label_list):
            for item in selected_label_list:
                glyph = lm.glyphs[lm.legend_labels.index(item)]
                glyph.visible = True
                
                p.legend.items = [LegendItem(label=legend_item, renderers=[lm.glyphs[lm.legend_labels.index(legend_item)]]) for idx, legend_item in enumerate(selected_label_list)]
                

        def update_grid_ranges(self, extent_array):
            self.grid_ranges = extent_array

    ## Variables
    lm = mutable_variables()

    grid_params = {"dimension": 0, "resolution": 500}
    # files and folders for local access
    fasta_list = [
        file for file in os.listdir() if os.path.isfile(file) and ".py" not in file
    ]
    model_list = [folder for folder in os.listdir() if os.path.isdir(folder)]

    VAE_custom_legend_pallet = [
        "#ee6222",
        "#eeb422",
        "#898f9b",
        "#ee22c2",
        "#6e5410",
        "#b422ee",
        "#ffe755",
        "#c3941c",
        "#ee4e22",
        "#ee22d6",
        "#54106e",
        "#8b22ee",
        "#f59a81",
        "#ee22ae",
        "#b2b349",
        "#7c7d56",
        "#28281b",
        "#b7d290",
        "#d2909c",
        "#d290c4",
        "#5f49b3",
        "#fcadc4",
        "#ab90d2",
        "#b34967",
        "#c2a0aa",
        "#49b395",
        "#f5ae3d",
        "#f5643d",
        "#c27b0a",
        "#fad69e",
        "#fab19e",
        "#d1ffff",
        "#6bffff",
        "#bf4f4f",
        "#685665",
        "#8c321c",
    ] + list(Set3[12])
    VAE_custom_legend_pallet = cycle(VAE_custom_legend_pallet)

    ## Functions

    def create_readme(dir_name: str) -> None:
        # creates training information and saves to directory after model training
        with open(os.path.join(dir_name, "model_creation.log"), "w") as f:
            f.writelines("VAE Training Start Time\n")
            f.writelines(model_params["train_start_time"] + "\n")
            f.writelines("VAE Training End Time\n")
            f.writelines(str(datetime.datetime.now()) + "\n\n")
            f.writelines([f"{x[0]} : {x[1]}\n" for x in model_params.items()])

    def train_model(model_dir_name: str, fasta_input, model_parameters: dict) -> None:
        # train model according to default parameters with fasta_input, save to output directory
        # tensorflow callbacks
        model_save_dir = os.path.join(model_dir_name, "trained_vae")
        earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=model_params["patience"]
        )
        save_best_model = tf.keras.callbacks.ModelCheckpoint(
            model_save_dir, monitor="loss", model="min", save_best_only=True
        )
        # model setup
        # num_sequences, seq_len = get_fasta_file_dimensions(fasta_input)
        model_parameters["num_seqs"] = lm.num_seqs
        model_parameters["seq_len"] = lm.seq_len
        model_parameters["hu_num"] = 3 * lm.seq_len
        vae = VAE(
            num_aa_types=23,
            dim_latent_vars=2,
            dim_msa_vars=lm.seq_len,
            num_hidden_units=lm.seq_len * 3,
            activation_func=model_parameters["activation"],
            regularization=model_parameters["l2_reg"],
        )
        vae.compute_output_shape(input_shape=(None, 23 * lm.seq_len))
        # setup dataset
        ds = tf.data.Dataset.from_generator(
            lambda: read_fasta_as_one_hot_encoded(fasta_input), tf.int8
        )
        ds = ds.shuffle(
            1000
        )  # Choose a random sequence from a buffer of 1000 sequences.
        ds = ds.batch(model_parameters["batch_size"])
        # train model!
        model_parameters["train_start_time"] = str(datetime.datetime.now())
        vae.compile(optimizer=tf.keras.optimizers.Adam())
        vae.fit(
            ds,
            epochs=model_parameters["epochs"],
            # validation_data=(test_msa, test_msa),
            callbacks=[earlystopping, save_best_model],
        )

    def get_axis_values(loaded_model: tf.keras.models.Model, fasta_input: str) -> int:
        ds = tf.data.Dataset.from_generator(
            lambda: read_fasta_as_one_hot_encoded(fasta_input), tf.int8
        ).batch(1)
        zed, _, _ = loaded_model.encoder.predict(ds)
        largest_value = abs(max(zed.min(), zed.max()))
        return int(np.ceil(largest_value))

    def get_key(val: int) -> str:
        for key, value in seq_code.items():
            if val == value:
                return key

    def return_sequence(latent_output: np.array) -> str:
        seq = "".join(get_key(x) for x in np.argsort(latent_output, axis=0)[-1, :])
        return seq

    def make_grid_msa(
        loaded_model: tf.keras.models.Model, batch_size=10000
    ) -> np.array:
        sampling_set = np.linspace(
            -grid_params["dimension"],
            grid_params["dimension"],
            grid_params["resolution"],
        )
        a = np.meshgrid(sampling_set, sampling_set)
        coord = np.vstack(np.array(a).transpose())
        with open("temp_landscape.fasta", "w") as fd:
            for batch_idx in range(0, coord.shape[0], batch_size):
                if batch_idx + batch_size > coord.shape[0]:  # bigger than array
                    z_input = coord[batch_idx:]
                else:
                    z_input = coord[batch_idx : batch_idx + batch_size]
                latent_output = loaded_model.decoder.predict(z_input)
                sequences = [return_sequence(seq_mat) for seq_mat in latent_output]
                for idx_seq, (x, y) in enumerate(z_input):
                    fd.writelines("> " + str(x) + " " + str(y) + "\n")
                    fd.writelines(sequences[idx_seq] + "\n")
        return coord

    def get_hamiltonian(dir_name: str, coords_for_pkl: np.array) -> np.array:
        mfdcamodel = dca(os.path.join(dir_name, "training_sequences.fasta"))
        mfdcamodel.mean_field()

        grid_hamiltonians, _ = mfdcamodel.compute_Hamiltonian("temp_landscape.fasta")

        output_grid = np.zeros((coords_for_pkl.shape[0], 3))
        output_grid[:, :2] = coords_for_pkl
        output_grid[:, 2] = grid_hamiltonians

        return output_grid

    def generate_landscape(dir_name: str) -> None:
        # Use model to find training data area, then plot a grid around that area.
        # Generates fasta file, score fasta file, create grid file, delete fasta file
        model_name = os.path.join(dir_name, "trained_vae")
        local_model = load_model(model_name, compile=True)
        # get dimensions
        grid_params["dimension"] = get_axis_values(
            local_model, os.path.join(dir_name, "training_sequences.fasta")
        )
        # create landscape fasta, get grid
        stacked_coords = make_grid_msa(local_model)
        # score landscape fasta, create final landscape visual file
        output_landscape = get_hamiltonian(dir_name, stacked_coords)
        # save landscape seqs as b64 and clean up
        with open(
            os.path.join(dir_name, "temp_landscape.fasta"), "rb"
        ) as fasta_file:
            landscape_b64 = base64.b64encode(fasta_file.read())
        os.remove("temp_landscape.fasta")
        # save input seqs as base64 obj
        with open(
            os.path.join(dir_name, "training_sequences.fasta"), "rb"
        ) as fasta_file:
            training_b64 = base64.b64encode(fasta_file.read())
        # write pickle to directory
        output_path = os.path.join(dir_name, "latent_generative_landscape.pkl")
        with bz2.BZ2File(output_path, "wb") as f:
            pickle.dump(
                [
                    model_name,
                    output_landscape,
                    lm.seq_len,
                    local_model.get_weights(),
                    training_b64,
                    landscape_b64,
                ],
                f,
            )
        print("LGL saved in model folder")

    # Color Functions
    def gencolor(numpoints: int) -> np.array:
        rgb = np.array(
            [
                [r, g, b]
                for r, g, b in zip(
                    np.random.randint(100, 150) + np.linspace(0, 105, numpoints),
                    np.random.randint(100, 150) + np.linspace(0, 105, numpoints),
                    np.random.randint(100, 150) + np.linspace(0, 105, numpoints),
                )
            ],
            dtype="uint8",
        )
        return [('#{:X}{:X}{:X}').format(x[0], x[1], x[2]).lower() for x in rgb]


    solid_cycler = cycle(Colorblind8)
    next(solid_cycler)

    ## Bokeh Event Functions

    def bokeh_create_model_folder(event) -> None:
        # use model_name_input text input to create folder
        dir_name = model_name_input.value
        if dir_name not in os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)),"docker")):
            os.mkdir(lm.train_model_name)
            info_text.text = (
                info_text.text
                + "\nModel directory created: "
                + str(model_name_input.value)
            )
        else:
            info_text.text = (
                info_text.text
                + "\nModel directory already taken, please choose another."
            )
            raise ValueError("Directory already taken")

    def select_fasta_for_training(attr, old, new) -> None:
        decoded = base64.b64decode(
            new
        )  # bokeh reads inputs in base64 so we need to decode
        bytes_decoded = io.BytesIO(decoded).read()  # this line decodes into bytes
        text_obj = bytes_decoded.decode("UTF-8")  # now we convert into an text_obj
        # Now we write this file to the model directory which has already been selected.
        fasta_location = os.path.join(lm.train_model_name, "training_sequences.fasta")
        with open(fasta_location, "w") as f:
            f.writelines(io.StringIO(text_obj).read())

        # lm.fasta_name = io.StringIO(
        #     text_obj
        # ).read()  # we convert text into StringIO and then read as str

        lm.num_seqs, lm.seq_len = get_fasta_file_dimensions(
            fasta_location
        )  # file dim are saved in dic
        lm.train_fasta = fasta_location
        info_text.text = (
            info_text.text + "\nFasta Selected!"
        )  # user feedback provided to window

    def select_model_name(attr, old, new) -> None:
        lm.train_model_name = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),"docker"), new)
        info_text.text = (
            info_text.text
            + "\nModel name: "
            + new
        )

    def bokeh_train_model(event) -> None:
        # train model, tell the user
        info_text.text = info_text.text + "\nModel is training, please wait..."
        train_model(lm.train_model_name, lm.train_fasta, model_params)
        # Save training information.
        create_readme(lm.train_model_name)
        info_text.text = info_text.text + "\nModel complete, creating LGL..."
        # Create landscape
        print("Creating latent generative landscape...")
        generate_landscape(lm.train_model_name)
        info_text.text = info_text.text + "\nLGL created, model is ready to use."

        # Dropdown no longer used for selection
        # Refresh folder list for model selection
        # new_model_list = [folder for folder in os.listdir() if os.path.isdir(folder)]
        # plot_model_selection.menu = new_model_list

    def bokeh_load_model(attr, old, new) -> None:
        lm.update_model(new)
        print(lm.the_model_name)
        info_text.text = info_text.text + "\nLoaded model for plotting."

    def plot_landscape(event) -> None:  # plots selected grid_dataset.pkl
        lm.bp_color_size = ["white", 3]
        p.x_range.range_padding = p.y_range.range_padding = 0
        p.grid.grid_line_width = 0.5
        grid_dataset = lm.landscape_grid

        # ✅ Sort by x (primary), then y (secondary) to ensure consistent ordering
        grid_dataset = grid_dataset[np.lexsort((grid_dataset[:,1], grid_dataset[:,0]))]

        pixels = grid_dataset[grid_dataset[:, 0] == grid_dataset[0, 0]].shape[0]
        lm.pixels = pixels
        image_grid = np.zeros((pixels, pixels))
        index_grid = np.zeros((pixels, pixels))
        count = 0
        for i in range(pixels):
            for j in range(pixels):
                image_grid[j, i] = grid_dataset[count][2]
                index_grid[j, i] = count
                count += 1
        lm.index_grid = np.flipud(np.rot90(index_grid))
        xmin, ymin = grid_dataset[0][0], grid_dataset[0][1]
        xmax, ymax = grid_dataset[-1][0], grid_dataset[-1][1]
        lm.update_grid_ranges([xmin, ymin, xmax, ymax])
        xspan, yspan = xmax - xmin, ymax - ymin
        output_plot = p.image(
            image=[image_grid], x=xmin, y=ymin, dw=xspan, dh=yspan, level="image"
        )
        color_mapper.update(low=grid_dataset[:, 2].min(), high=grid_dataset[:, 2].max())
        output_plot.glyph.color_mapper = color_mapper
        lm.update_grid_plot(output_plot)
        color_bar.visible = True


    def plot_base_data(event) -> None:  # plots primary dataset, selectable by lasso
        newds = tf.data.Dataset.from_generator(
            lambda: read_fasta_as_one_hot_encoded(lm.train_fasta), tf.int8
        )
        newds = newds.batch(1)
        newlatent, _, _ = lm.the_model.encoder.predict(newds)
        # build CDS
        lm.train_fasta.seek(0)

        newheaders = []
        newseqs = []
        for x in SeqIO.parse(lm.train_fasta, "fasta"):
            newheaders.append(x.description)
            newseqs.append(str(x.seq))
        new_data_dictionary = {"Name": newheaders, "Sequence": newseqs}
        lm.train_fasta.seek(0)
        for dimension in range(newlatent.shape[1]):
            new_data_dictionary["z" + str(dimension)] = newlatent[:, dimension]
        new_data_dictionary["colors"] = [
            lm.bp_color_size[0] for _ in range(newlatent.shape[0])
        ]
        new_data_dictionary["Labels"] = [
            "Training Data" for _ in range(newlatent.shape[0])
        ]

        new_df = pd.DataFrame(data=new_data_dictionary)
        lm.df = pd.concat([lm.df, new_df], ignore_index=True)
        # Initialize the main DataFrame and ColumnDataSource
        lm.base_cds.data.update(lm.df)
        
        # plot glyph using view to filter for training data
        base = p.scatter(
            "z0",
            "z1",
            fill_color="colors",
            line_color=None,
            size=lm.bp_color_size[1],
            legend_label="Training Data",
            source=lm.base_cds,
            muted_alpha=0.2,
            level="glyph",
            view=CDSView(source=lm.base_cds, filters=[GroupFilter(column_name='Labels', group='Training Data')])
        )
        lm.update_base(base)
        lm.glyphs.append(base)
        lm.update_legend_labels('Training Data')
        lm.base_cds.selected.on_change("indices", select_points)
        p.legend.click_policy = "mute"
        if lm.bp_color_size[0] == "steelblue":
            p.x_range.range_padding = p.y_range.range_padding = 0.75
        select_seqs_to_relabel.options = lm.legend_labels
        lm.recolor_label = "Training Data"
        update_checkbox()

    def plot_data(attr, old, new) -> None:  # plots any additional datapoints
        decoded = base64.b64decode(new)  
        bytes_decoded = io.BytesIO(decoded).read()  
        text_obj = bytes_decoded.decode("UTF-8")  
        fasta_in = io.StringIO(text_obj)  
        newds = tf.data.Dataset.from_generator(
            lambda: read_fasta_as_one_hot_encoded(fasta_in), tf.int8
        )
        newds = newds.batch(1)
        newlatent, _, _ = lm.the_model.encoder.predict(newds)
        fasta_in.seek(0)
        newheaders = []
        newseqs = []
        for x in SeqIO.parse(fasta_in, "fasta"):
            newheaders.append(x.description)
            newseqs.append(str(x.seq))
        new_data_dictionary = {"Name": newheaders, "Sequence": newseqs}
        for dimension in range(newlatent.shape[1]):
            new_data_dictionary["z" + str(dimension)] = newlatent[:, dimension]
        new_data_dictionary["Labels"] = [
            add_file_name for _ in range(newlatent.shape[0])
        ]
        # plot solid or gradient colors
        if lm.plot_is_gradient:
            new_data_dictionary["colors"] = gencolor(newlatent.shape[0])
        else:
            chosen_color = next(solid_cycler)
            new_data_dictionary["colors"] = [
                chosen_color for _ in range(newlatent.shape[0])
            ]
        
        # Add new data to existing DataFrame
        new_df = pd.DataFrame(data=new_data_dictionary)
        lm.df = pd.concat([lm.df, new_df], ignore_index=True)
        
        # Update the CDS with complete DataFrame
        lm.base_cds.data.update(lm.df)
        
        # Add new glyph using same CDS but with view filter
        base = p.scatter(
                "z0",
                "z1",
                fill_color="colors",
                line_color=None,
                size=lm.bp_color_size[1],
                legend_label=add_file_name,
                source=lm.base_cds,
                muted_alpha=0.2,
                level="glyph",
                view=CDSView(source=lm.base_cds, filters=[GroupFilter(column_name='Labels', group=add_file_name)])
            )
        lm.update_base(base)
        lm.glyphs.append(base)
        lm.legend_labels.append(add_file_name)
        
        # Update legend and checkbox
        select_seqs_to_relabel.options = lm.legend_labels
        p.legend.location = "top_left"
        p.legend.click_policy = "mute"
        update_checkbox()

    def set_seq_to_recolor(attr, old, new) -> None:
        lm.recolor_label = new

    def select_points(attr, old, new)-> None:  # outputs sequences of lasso selection to textbox
        temp_df = lm.df
        temp_df = temp_df.reset_index(drop=True)
        temp_df = temp_df.loc[new]
        fasta.text = "\n".join(
            [
                str(">" + x + "\n")
                for x in temp_df["Name"]
            ]
        )

    def select_map_points(event): #selects sequences from hamiltonian map plot
        geo = event.geometry
        xx = geo['x']; yy = geo['y']
        gridline_x = np.linspace(lm.grid_ranges[0],lm.grid_ranges[2],lm.pixels)
        gridline_y = np.linspace(lm.grid_ranges[1],lm.grid_ranges[3],lm.pixels)
        cl = np.arange(lm.pixels)
        step_size_x = gridline_x[1] - gridline_y[0]
        step_size_y = gridline_y[1] - gridline_y[0]
        x_grid = [cl[np.isclose(xx[x],gridline_x,atol=step_size_x)][0] for x in range(len(xx))]
        y_grid = [cl[np.isclose(yy[x],gridline_y,atol=step_size_y)][0] for x in range(len(yy))]
        rr,cc = polygon(x_grid,y_grid,shape=(lm.pixels,lm.pixels))
        selected_points = lm.index_grid[rr,cc].astype(np.int64)
        lm.selected_positions = selected_points
        fasta.text ='New sequences selected!'

    def save_landscape_seqs(event):
        print('saving landscape selection...')
        count = 0
        # progress.text = '\tz0  \t\t  z1\n'
        grid_seq_parser = SeqIO.parse(lm.grid_seq_location,'fasta')
        with open('landscape_selection.fasta','w') as fd:
            for sequence in grid_seq_parser:
                if count in lm.selected_positions:
                    fd.writelines('>'+sequence.description+'\n'+sequence.seq+'\n')
                    fasta.text = fasta.text + sequence.description + '\n'
                count+=1
        # progress.text = 'Done!\n'+progress.text

    def change_plot_type(
        event,
    ) -> None:  # switch between gradient and solid color for additional data
        if lm.plot_is_gradient:
            lm.plot_is_gradient = False
            color_choice.label = "Plot Additional MSA as Solid Color"
        else:
            lm.plot_is_gradient = True
            color_choice.label = "Plot Additional MSA as Gradient"

    def select_data_csv(
        attr, old, new
    ) -> None:  # loads csv, updates column select dropdown
        decoded = base64.b64decode(
            new
        )  # bokeh reads inputs in base64 so we need to decode
        bytes_decoded = io.BytesIO(decoded)
        lm.update_labeldf(pd.read_csv(bytes_decoded))
        lm.update_ldf_labels(lm.label_df)
        column_select.menu = lm.ldf_labels

    def update_checkbox() -> None:  # used to initialize new data in Legend tab
        the_checkbox.labels = [str(x) for x in set(lm.df["Labels"])]
        the_checkbox.active = list(range(len(set(lm.df["Labels"]))))

    def change_training_colors(event) -> None:
        # two types of data will be plotted, categorical and numerical
        # detect which type (based on #unique/#datapoints)
        # if categorical, define color column based on class
        # else, plot with colormap.
        decision_ratio = len(lm.label_df[event.item].unique()) / len(
            lm.label_df[event.item]
        )

        base_tooltip = p.hover[0].tooltips
        if len(base_tooltip) > 1:
            base_tooltip.pop(-1)
        base_tooltip.append((event.item + " ", "@" + event.item))
        p.hover[0].tooltips = base_tooltip

        if decision_ratio >= 0.7:
            cm = p.select_one(LinearColorMapper)
            cm.update(
                low=lm.label_df[event.item].min(), high=lm.label_df[event.item].max()
            )
            lm.base_plot.glyph.fill_color = {
                "field": event.item,
                "transform": color_mapper,
            }

            color_bar.visible = True
            update_checkbox()
        else:
            if lm.bp_color_size[0] == "white":
                color_bar.visible = True
            value_list = lm.label_df[event.item].unique().tolist()
            color_list = [
                next(VAE_custom_legend_pallet) for _ in range(len(value_list))
            ]
            newcolors = {v: c for c, v in zip(color_list, value_list)}
            colored_values = [newcolors[x] for x in lm.label_df[event.item]]
            # update data
            lm.update_cds_column(event.item, colored_values)

            lm.base_cds.data.update(lm.df)

            # Reset glyphs and legend
            for glyph in lm.glyphs:
                if glyph in p.renderers:
                    p.renderers.remove(glyph)
            lm.glyphs = []
            p.legend.items =[]
    
            # Create new glyph for each unique label using filtered views
            lm.legend_labels = list(set(lm.df['Labels']))
            for label in lm.legend_labels:
                view = CDSView(source=lm.base_cds, filters=[GroupFilter(column_name='Labels', group=label)])
                glyph = p.scatter(
                        "z0",
                        "z1",
                        source=lm.base_cds,
                        view=view,
                        fill_color="colors",
                        line_color=None,
                        legend_label=label,
                        muted_alpha=0.2,
                        size=lm.bp_color_size[1],
                        level="glyph"
                )
                p.renderers.append(glyph)
                lm.glyphs.append(glyph) # add glyph to list
            # lm.update_labels(list(set(lm.label_df[event.item])))
            update_checkbox()

    def toggle_legend(event) -> None:  # turn off/on legend
        if p.legend.visible == True:
            p.legend.visible = False
            leg.label = "Turn On Legend"
        else:
            p.legend.visible = True
            leg.label = "Turn Off Legend"

    def checkall(event) -> None:  # check all functionality for legend
        if len(the_checkbox.active) == len(set(lm.df["Labels"])):
            the_checkbox.active = []
        else:
            the_checkbox.active = list(
                range(len(set(lm.df["Labels"])))
            )

    def update_checkbox_data(
        attr, old, new
    ) -> None:  # updates base glyph data with selected data.
        active_labels = [the_checkbox.labels[i] for i in the_checkbox.active]
        labels_to_remove = [
            the_checkbox.labels[i]
            for i in range(len(the_checkbox.labels))
            if i not in the_checkbox.active
        ]
        lm.add_from_selection(active_labels)
        if labels_to_remove:
            lm.remove_from_selection(labels_to_remove)
    
    def save_selected_fasta(event):
        print("Saving selected sequences..")
        if len(lm.base_cds.selected.indices) > 0:
            selected_df = lm.df.iloc[lm.base_cds.selected.indices]
            print(f"Saving {len(selected_df)} selected sequences...")
        else:
            selected_df = lm.df
            print(f"No selection - saving all {len(selected_df)} sequences...")
        
        with open('landscape_selection.fasta','w') as fd:
            for row in range(0,selected_df.shape[0]):
                fd.write(">"+selected_df.iloc[row]["Name"]+"\n"+selected_df.iloc[row]["Sequence"]+"\n")
        print("DONE")
        fasta.text = f"Saved {len(selected_df)} sequences to landscape_selection.fasta"

    def generate_drawn_points(event):
        points = [[gen_source.data['x'][idx], gen_source.data['y'][idx]] for idx in range(len(gen_source.data['x']))]
        seq_mats = lm.the_model.decoder.predict(points)
        sequences = [return_sequence(seq) for seq in seq_mats]
        with open('generated_sequences.fasta','w') as fd:
            for idx in range(len(sequences)):
                fd.write(">"+str(points[idx])+"\n"+sequences[idx]+"\n")
        return

    # Plotting Component

    title = Title()
    p = figure(
        title=title, x_axis_label="z0", y_axis_label="z1", tools="pan,wheel_zoom,lasso_select,reset,save", toolbar_location="below", output_backend=OUTPUT_BACKEND
    )

    
    # p.on_event("selectiongeometry", select_map_points)
    
    # Hover information
    p.add_tools(HoverTool(tooltips=[("ID ", "@Name")]))

    # Add point generation
    gen_source = ColumnDataSource(data=dict(x=[], y=[]))
    gen_render = p.circle('x', 'y', source=gen_source, size=8, line_color="black", level="overlay")
    point_draw_tool = PointDrawTool(renderers=[gen_render])
    p.add_tools(point_draw_tool)


    # Define colorbar for landscape/plotting, hide initially
    color_mapper = LinearColorMapper(palette=Viridis256, low=0, high=1)
    color_bar = ColorBar(color_mapper=color_mapper, width=5, label_standoff=5)
    color_bar.visible = False
    p.add_layout(color_bar, "right")

    ## Tab One Components

    # text - file_input - train_button - folder_input - load_button
    info_text = PreText(
        text="""Usage:\n1.) Create a model directory then select a fasta for model training.\n2.) Click "Train model"\nWhen/if model has been    trained:\n1.) Select LGL pickle file\n2.) Click "Plot LGL", and continue to next tab. """,
        style={"height": "300px", "width": "600px"},
    )

    # training user inputs and update calls
    model_info = PreText(text="Input model name:")
    model_name_input = TextInput(
        placeholder="Model name, do not use spaces.",
        value="",
    )
    model_name_input.on_change("value", select_model_name)

    # Create model folder button
    model_folder_create = Button(label="Create model folder", button_type="success")
    model_folder_create.on_click(bokeh_create_model_folder)

    # File input
    fasta_selection = FileInput()
    fasta_title = Div(text="Select aligned fasta:")
    fasta_selection.on_change("value", select_fasta_for_training)

    # train button
    train = Button(label="Train VAE model", button_type="success")
    train.on_click(bokeh_train_model)

    # folder input for plotting
    model_selection_title = Div(
        text="Select latent_generative_landscape.pkl from model directory to load LGL for plotting:",
        align="center",
    )
    plot_model_selection = FileInput()
    plot_model_selection.on_change("value", bokeh_load_model)
    # Landscape plot button
    plot_lgl = Button(label="Plot LGL", button_type="success")
    plot_lgl.on_click(plot_landscape)
    # Build layout
    panel_one_layout = column(
        info_text,
        model_info,
        row(model_name_input, model_folder_create),
        fasta_title,
        fasta_selection,
        train,
        model_selection_title,
        plot_model_selection,
        plot_lgl,
    )
    panel_one = Panel(child=panel_one_layout, title="Model Training")

    ## Tab Two Components
    # Instruction - PlotTrainingData - PlotAdditional -
    # ColorAdditional - LoadCSV - CSV Column - Toggle Legend - Selected Seqs

    # Instruction
    instruct_text = PreText(
        text="""Plotting functions:
        -Select "Plot training data" to encode your input sequences
        into the landscape.
        -Select "Plot Additional MSA as..." to color additional sequences
        as a gradient or as a solid color
        -Select "Additional Sequences" to plot other fasta files. 
        -To relabel plotted sequences using a csv, select a group to relabel.
        Select "Load CSV" to load a CSV, where the first row is column labels,
        with rows corresponding to MSA sequences.
        -Select "Choose CSV column" to recolor data according to
        your class label.
        -Select "Toggle Legend" to turn off the legend.
        Next tab allows you to remove sequences with specific labels
        from the plot.
        """,
        style={"height": "300px", "width": "600px"},
    )
    # PlotTrainingData
    msa_d = Button(label="Plot training data", button_type="success")
    msa_d.on_click(plot_base_data)

    # PlotAdditional
    def passy(attr, old,new):
        global add_file_name
        add_file_name = new
        return

    myFileNames = TextInput(value="", title="File names:")
    myFileValues = TextInput(value="", title="File values:")

    add_seq_title = Div(text="Additional Sequences")
    add_d = FileInput(multiple=True)
    add_d.js_on_change("value", CustomJS(args=dict(myFileNames=myFileNames, myFileValues=myFileValues), code="""
    myFileNames.value = this.filename.toString();
    myFileValues.value = this.value.toString();
"""))
    myFileNames.on_change("value", passy)
    myFileValues.on_change("value", plot_data)


    # ColorAdditional
    color_choice = Button(
        label="Plot Additional MSA as Gradient", button_type="success"
    )
    color_choice.on_click(change_plot_type)

    # Dropdown to select which group to relabel
    select_seqs_to_relabel = Select(title="Select sequences to relabel using CSV", options=lm.legend_labels)
    select_seqs_to_relabel.on_change("value",set_seq_to_recolor)

    # LoadCSV
    add_seq_csv = Div(text="Load CSV")
    csv_select = FileInput()
    csv_select.on_change("value", select_data_csv)

    # CSV Column
    column_select = Dropdown(label="Choose CSV column", menu=lm.ldf_labels)
    column_select.on_click(change_training_colors)

    # Toggle Legend
    leg = Button(label="Turn Off Legend", button_type="success")
    leg.on_click(toggle_legend)

    # Selected Seqs
    fasta = Button(label="Save selected sequences as fasta")
    fasta.on_click(save_selected_fasta)

    # Generate Seqs
    gen_button = Button(label="Generate sequences from points drawn")
    gen_button.on_click(generate_drawn_points)


    # Collect into panel for tab
    panel_two_layout = column(
        instruct_text,
        msa_d,
        add_seq_title,
        color_choice,
        add_d,
        select_seqs_to_relabel,
        add_seq_csv,
        csv_select,
        column_select,
        leg,
        fasta,
        gen_button
    )
    panel_two = Panel(child=panel_two_layout, title="Plotting Tools")

    ## Tab Three Components
    # Toggle All - Checkbox
    checkbox_toggle = Button(label="Un/Check All", button_type="success")
    checkbox_toggle.on_click(checkall)

    # Checkbox
    the_checkbox = CheckboxGroup(labels=[])
    the_checkbox.on_change("active", update_checkbox_data)

    # collect into panel for tab
    panel_three_layout = column(checkbox_toggle, the_checkbox)
    panel_three = Panel(child=panel_three_layout, title="Legend Editing")

    ## Tab Four Components - 3D Landscape
    def create_3d_plot():
        if lm.landscape_grid is not None:
            try:
                grid_dataset = lm.landscape_grid
                
                # Extract x, y, z values for the surface
                x = grid_dataset[:, 0]
                y = grid_dataset[:, 1]
                z = grid_dataset[:, 2]
                
                # Determine grid dimensions
                x_unique = np.unique(x)
                y_unique = np.unique(y)
                nx = len(x_unique)
                ny = len(y_unique)
                
                # Reshape z into a grid
                Z = np.zeros((ny, nx))
                for i, xi in enumerate(x_unique):
                    for j, yi in enumerate(y_unique):
                        idx = np.where((x == xi) & (y == yi))[0]
                        if len(idx) > 0:
                            Z[j, i] = z[idx[0]]
                
                # Create the figure with the surface
                fig = go.Figure(data=[go.Surface(z=Z, x=x_unique, y=y_unique, opacity=0.8)])
                
                # Add scatter points if we have data in the DataFrame
                if not lm.df.empty and 'z0' in lm.df.columns and 'z1' in lm.df.columns:
                    # Get unique labels for coloring
                    unique_labels = lm.df['Labels'].unique()
                    
                    # For each label group, add a scatter3d trace
                    for label in unique_labels:
                        df_subset = lm.df[lm.df['Labels'] == label]
                        
                        # Get z values for these points by interpolating from the surface
                        scatter_z = []
                        for idx, row in df_subset.iterrows():
                            # Find closest grid point
                            x_idx = np.abs(x_unique - row['z0']).argmin()
                            y_idx = np.abs(y_unique - row['z1']).argmin()
                            # Get z value from grid
                            scatter_z.append(Z[y_idx, x_idx])
                        
                        # Add scatter points for this label group
                        fig.add_trace(go.Scatter3d(
                            x=df_subset['z0'],
                            y=df_subset['z1'],
                            z=scatter_z,
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=df_subset['colors'].iloc[0] if len(set(df_subset['colors'])) == 1 else df_subset['colors'],
                                opacity=0.8
                            ),
                            name=label,
                            text=[f"ID: {name}<br>Sequence: {seq[:20]}..." for name, seq in zip(df_subset['Name'], df_subset['Sequence'])],
                            hoverinfo='text',
                            hovertemplate='%{text}<br>z0: %{x:.4f}<br>z1: %{y:.4f}<br>Energy: %{z:.4f}<extra></extra>'
                        ))
                
                # Update layout
                fig.update_layout(
                    title='3D Latent Generative Landscape with Data Points',
                    scene=dict(
                        xaxis_title='z0',
                        yaxis_title='z1',
                        zaxis_title='Energy',
                        camera=dict(
                            up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=1.5, y=1.5, z=1.5)
                        )
                    ),
                    width=800,
                    height=600,
                    margin=dict(l=0, r=0, b=0, t=30)
                )
                
                # Save to a temporary HTML file and set up server as before
                import tempfile
                import os
                import threading
                import http.server
                import socketserver
                
                # Create a temporary file with a predictable name
                temp_dir = tempfile.gettempdir()
                temp_file = os.path.join(temp_dir, "vae_3d_plot.html")
                
                # Write the plot to the file
                import plotly.io as pio
                with open(temp_file, 'w') as f:
                    f.write(pio.to_html(fig, full_html=True, include_plotlyjs='cdn'))
                
                print(f"Saved 3D plot to: {temp_file}")
                
                # Set up a simple HTTP server in a separate thread
                PORT = 8000
                DIRECTORY = temp_dir
                
                class Handler(http.server.SimpleHTTPRequestHandler):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, directory=DIRECTORY, **kwargs)
                
                def start_server():
                    with socketserver.TCPServer(("", PORT), Handler) as httpd:
                        print(f"Serving at port {PORT}")
                        httpd.serve_forever()
                
                # Start the server in a separate thread if it's not already running
                try:
                    import socket
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect(("localhost", PORT))
                    s.close()
                    print("Server already running")
                except:
                    print("Starting server")
                    server_thread = threading.Thread(target=start_server, daemon=True)
                    server_thread.start()
                
                # Create an iframe that points to the local server
                plot_filename = os.path.basename(temp_file)
                iframe_html = f"""
                <iframe src="http://localhost:{PORT}/{plot_filename}" width="800" height="600" frameborder="0"></iframe>
                <p>If the plot is not visible, you can open it directly at: 
                <a href="http://localhost:{PORT}/{plot_filename}" target="_blank">http://localhost:{PORT}/{plot_filename}</a></p>
                """
                
                return iframe_html
                
            except Exception as e:
                print(f"Error creating 3D plot: {e}")
                import traceback
                traceback.print_exc()
                return f"<p>Error creating 3D plot: {str(e)}</p>"
        return "<p>Please load a model first</p>"

    # Create HTML component for 3D plot
    plot_3d = Div(text="<p>Click 'Update 3D Plot' to generate the visualization</p>", width=800, height=600)
    
    # Button to update 3D plot
    update_3d = Button(label="Update 3D Plot", button_type="success")
    def update_3d_plot(event):
        plot_3d.text = create_3d_plot()
    update_3d.on_click(update_3d_plot)

    # Collect into panel for tab
    panel_four_layout = column(update_3d, plot_3d)
    panel_four = Panel(child=panel_four_layout, title="3D Landscape")

    # Build and run - modify this part to conditionally show plots
    tabs = Tabs(tabs=[panel_one, panel_two, panel_three, panel_four])
    
    # Create a callback to hide/show the main plot based on active tab
    def update_visibility(attr, old, new):
        if new == 3:  # If 3D Landscape tab is selected (index 3)
            p.visible = False  # Hide the main plot
            plot_3d.width = 1600  # Make 3D plot wider to cover the space
            plot_3d.height = 800  # Make 3D plot taller
        else:
            p.visible = True  # Show the main plot for other tabs
            plot_3d.width = 800  # Reset 3D plot size
            plot_3d.height = 600
    
    # Register the callback
    tabs.on_change('active', update_visibility)
    
    # Add both to the document in a layout that allows conditional visibility
    doc.add_root(column(row(p, tabs)))


if __name__ == "__main__":
    args = parse_args()
    
    # Create a global variable to store the output backend
    global OUTPUT_BACKEND
    OUTPUT_BACKEND = "svg" if args.save_as_svg else "canvas"
    
    print("Opening Bokeh application on http://localhost:5006/")
    server = Server(
        {"/": vae_lgl_analysis_app}, 
        port=5006, 
        websocket_max_message_size=1000000000000, 
        allow_websocket_origin=["localhost:5006","5006"]
    )
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
