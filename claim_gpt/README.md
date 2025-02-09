# How to run notebook claim_gpt.ipynb

I don't want notebooks to be run in place since this will trigger a github update.

Instead, do the following:

* Create a folder under ClaimGPT250203 that begins with the prefix 'home_' (for example, home_claim)
* .gitignore ignores folders that start with home_
* Copy claim_gpt.ipynb to that folder
* I use JupyterLab panel from Anaconda-Navigator to run the notebook.

# How to run other notebooks in the claim_gpt folder

Do the following:

* Copy the whole notebooks folder to your 'home_xxxxx' folder that you created above.
* Run one of the notebooks. The notebook should be set up to work correctly if the above is followed.

# How to run Python code

I don't want python code to be run in place since this will trigger a github update.

You can follow the method above for running Jupyter notebooks.
That is, create a folder that begins with the prefix 'home_' (for example, home_main_python).

In this way, you can debug some code using PyCharmCE for example.

I didn't test this method out, but it should work.