<p align="center">
  <h3 align="center">Correlation based Embedding Novelty Detection (CEND) </h3>

  <p align="center">
    Detecting words associated to a slow emerging topic
    <br />
    <br />
    <a href="https://arxiv.org/"><strong>Associated publication »</strong></a>
    <br />
  </p>
</p>



<!-- Sommaire -->
<details open="open">
  <summary><h2 style="display: inline-block">Summary</h2></summary>
  <ol>
    <li>
      <a href="#utilisation">Use</a>
      <ul>
        <li><a href="#Librairies">Libraries</a></li>
        <li><a href="#paramètres">Parameters</a></li>
        <li><a href="#util">Launch</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>




<!-- Utilisation -->
## Librairies

Libraries used for this model are listed in the:
``
requirements.txt
``

Use Python 3.6

### Parameters
2 parameters are used when launching ``cend.py``: 
``
--category and --model
`` 

``category`` is useful when you are using evaluation mode, it means, using a category that has been artificially introduced in the dataset. The new dataset is generated by
``simulation/simulation.py``
and stored in 
``scenarios/emergent_%category.csv``
If this parameter is empty, the model run on real data: the data path can be modified directly in the script at variable ``name``.


``model`` is used to change the type of embedding modelisation that we are using: it can be ``SGNS`` or ``SVD``
 permet de changer le type de modèle de représentation qu'on utilise pour représenter les mots.

Other variables can be set in the code

``LANGUAGE``:
``french`` 
``english`` . It is useful if you have different preprocessing functions depending on your language.

``size_window_corr``: size of the sliding window to calculate correlation. Correlation computing begin once the number of time step is greater than this value.

``size_init``: number of time steps used to initialized the first embedding model.


### Arborescence

1. Launch CEND
   ```sh
   python cend.py --category="restaurants" --model="SVD"
   ```

```
.
├── README.md
├── arrays # Movements and Correlations directory
├── cend.py # Main script
├── data # Data directory
├── models # Models directory (.model for SGNS, .npy for SVD)
├── preprocess.py # Fonctions de nettoyages des données
├── requirements.txt # Liste des librairies
├── scenarios # Simulated data directory
├── simulation
│   └── simulation.py # Script for data simulation

```

<!-- CONTACT -->
## Contact

Clément Christophe - clement.christophe@edf.fr




