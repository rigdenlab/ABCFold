<div class="tab-pane fade show" id="offline" role="tabpanel" aria-labelledby="offline-tab">
    <p>
        If you would like to use the PAE Viewer offline in your browser and
        start it with a command line interface, you can use the scripts
        provided by the repository
        (<a href="https://gitlab.gwdg.de/general-microbiology/pae-viewer" target="_blank"
    >general-microbiology/pae-viewer</a>). Python >=3.9 is required to run
        the offline version.
    </p>

    <h5>Instructions</h5>
    <div>
        1. Download the PAE Viewer project files
        (<a href="https://gitlab.gwdg.de/general-microbiology/pae-viewer/-/archive/main/pae-viewer-main.zip"
    >pae-viewer-main.zip</a>) and extract the archive.
    </div>

    <div>
        <p>
            2. Using the terminal, change the current working directory to
            the project root
            (<span class="font-monospace">/your/download/path/pae-viewer-main</span>)
            and start a local HTTP server with Python.
        </p>

        <div class="card" style="margin-bottom: 1em">
            <div class="card-body font-monospace small">
                cd /your/download/path/pae-viewer-main<br/>
                python -m http.server 8000
            </div>
        </div>
    </div>

    <div>
        <p>
            3. Open
            <a href="http://localhost:8000/standalone/pae-viewer.html" target="_blank">
                http://localhost:8000/standalone/pae-viewer.html
            </a>
            in your browser to start the offline version.
        </p>
    </div>

    <div>
        <p>
            4. To start PAE Viewer in your browser with arguments
            supplied via CLI, run the
            <span class="font-monospace">pae-viewer-main/standalone/pae_viewer.py</span>
            script from the project directory. For example:
        </p>

        <div class="card" style="margin-bottom: 1em">
            <div class="card-body font-monospace small">
                cd /your/download/path/pae-viewer-main/standalone<br/>
                python pae_viewer.py \<br/>
                &nbsp;&nbsp; --structure pae-viewer/sample/GatA-GatB/fold_gata_gatb_model_0.cif \<br/>
                &nbsp;&nbsp; --labels 'GatA;GatB' \<br/>
                &nbsp;&nbsp; --scores pae-viewer/sample/GatA-GatB/fold_gata_gatb_full_data_0.json \<br/>
                &nbsp;&nbsp; --crosslinks pae-viewer/sample/GatA-GatB/GatA-GatB.csv <br/>
            </div>
        </div>

        <p>
            For this, the local HTTP server must be running, as well. The Python
            script creates a HTML session file with all data embedded and opens
            it in the browser. It basically prefills the upload form and submits
            it. The session files are stored permanently in the
            <span class="font-monospace">pae-viewer-main/standalone</span>
            directory, and can be revisited at any time.
        </p>
    </div>
</div>
