

<div class="card mb-3">


  <div class="card-header">

    <ul class="nav nav-tabs card-header-tabs" id="tabs" role="tablist">

      <li class="nav-item" role="presentation">

        <button class="nav-link active" id="examples-tab" data-bs-toggle="tab"
                data-bs-target="#examples" type="button" role="tab"
                aria-controls="upload" aria-selected="true">
          Examples
        </button>

      </li>
      <!--
      <li class="nav-item" role="presentation">
        <button class="nav-link" id="upload-tab" data-bs-toggle="tab"
                data-bs-target="#upload" type="button" role="tab"
                aria-controls="upload" aria-selected="false">
          Upload
        </button>
      </li>

      <li class="nav-item" role="presentation">
        <button class="nav-link" id="offline-tab" data-bs-toggle="tab"
                data-bs-target="#offline" type="button" role="tab"
                aria-controls="offline" aria-selected="false">
          Offline version
        </button>
      -->
      </li>
      <li class="nav-item" role="presentation">
          <h2 id="header-title">ABCFold Results</h2>
      </li>

    </ul>
  </div>

  <div class="tab-content p-3" id="tabContents">

    {{paeViewerDemo__tab__examples.tpl}}
    {{paeViewerDemo__tab__upload.tpl}}
    {{paeViewerDemo__tab__offline.tpl}}
    {{paeViewerDemo__tab__citation.tpl}}
  </div>
</div>
</div>
