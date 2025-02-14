<div class="accordion-item">
  <h2 class="accordion-header" id="helpPtmHeading">
    <button class="accordion-button collapsed" type="button"
            data-bs-toggle="collapse" data-bs-target="#helpPtm"
            aria-expanded="true" aria-controls="helpPtm">
      <span class="fw-bold">PTMs</span>
    </button>
  </h2>
  <div id="helpPtm" class="accordion-collapse collapse"
       aria-labelledby="helpPtmHeading"
       data-bs-parent="#help">
    <div class="accordion-body">
      <div>
        <p>
          Post-translational modifications (PTMs) like the ones output
          by the AlphaFold Server are also supported by the PAE Viewer.
          Glycan chains will appear as separate chains with atom-wise
          PAE values. AlphaFold also outputs atom-wise PAE values for
          amino acids with different modifications. However, when uploading
          structures with these PTMs to PAE Viewer, the atom-wise values
          are replaced by their arithmetic means to get a residue-wise PAE and
          ensure consistent handling of residues. Note that for this reason,
          the PAE matrix displayed by the PAE Viewer will differ from
          the one on the AlphaFold web server, which still contains
          all atom-wise PAE values.
        </p>
      </div>
    </div>
  </div>
</div>
