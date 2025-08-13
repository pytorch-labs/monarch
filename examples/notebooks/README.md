# Example Notebooks

This folder contains some basic examples of using the Monarch API in jupyter notebooks.

## Setup
1. Follow the instructions outlined in ../../monarch/README.md to setup Monarch
2. Pip install jupyter:
    `pip install jupyter notebook`
3. Run your jupyter notebook: `jupyter notebook`
4. (optiona) In remote settings (as in a devserver), you can also port forward your jupyter notebook to your local machine. e.g.
    ```
    # devserver
    jupyter notebook --no-browser --port=8098

    #local
    ssh -N -L 8098:localhost:8098 <devserver_address>
````
5. Open localhost:8098 in your browser to see the jupyter notebook


## Manifest
* ping_pong.ipynb - Simple hello world with Actor API + Inter Actor Communication
