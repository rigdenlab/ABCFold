/*
data attribute creates object that is bound to this

computed - computed property but with caching
method - computed property - invoked each time

v-model - links inputs to vue js data


https://codepen.io/pespantelis/pen/ojwgPB
https://www.raymondcamden.com/2018/02/08/building-table-sorting-and-pagination-in-vuejs

*/

/* EventBus is used to pass changes between components */
// const EventBus = new Vue();

Vue.filter("decimalPlaces", (value, num = 2) => {
    if (value == null) {
        return "N/A";
    } else {
        return value.toFixed(num);
    }
});

Vue.component('abc-table', {
    data: function () {
        if (typeof window.sequence === 'undefined') {
            window.sequence = this.$root.sequence;
        }
        return {
            abc_models: this.$root.abc_models,
        }
    },
    mounted() {
        let show = Object.keys(this.$root.abc_models).length !== 0;
        toggleDisplay("abc-title", show);
        toggleDisplay("div1", show);


        if (show) {
            this.$nextTick(() => {
                // Ensure sequence, ft1, and ABC_rowFeatureMap are defined and accessible
                if (typeof sequence !== 'undefined' && typeof ft1 !== 'undefined' && typeof ABC_rowFeatureMap !== 'undefined') {
                    sortTableAndFeatures('abc_table', sequence, ft1, ABC_rowFeatureMap, '#div1', 2);
                } else {
                    console.error("Required variables (sequence, ft1, ABC_rowFeatureMap) are not defined.");
                }
            });
        }
      },
      methods: {
        getButtonClass(modelSource) {
            switch (modelSource) {
                case 'AlphaFold3':
                    return 'btn-source1';
                case 'Boltz-1':
                    return 'btn-source2';
                case 'Chai-1':
                    return 'btn-source3';
                default:
                    return 'btn-default';
            }
        }
    },
    template: `
    <div id="abc-table-container">
        <table id="abc_table">
            <thead>
                <tr>
                    <th title="The name of the model"
                        onclick="sortTableAndFeatures('abc_table', sequence, ft1, ABC_rowFeatureMap, '#div1', 0)">Model Name</th>
                    <th title="The source of the model"
                        onclick="sortTableAndFeatures('abc_table', sequence, ft1, ABC_rowFeatureMap, '#div1', 1)">Model Source</th>
                    <th title="The average pLDDT score of the model"
                        onclick="sortTableAndFeatures('abc_table', sequence, ft1, ABC_rowFeatureMap, '#div1', 2)">Average pLDDT</th>
                    <th title="The H-score of the model"
                        onclick="sortTableAndFeatures('abc_table', sequence, ft1, ABC_rowFeatureMap, '#div1',  3)">H-score</th>
                    <th title="The number of possible residue clashes found in the model - lower is better"
                        onclick="sortTableAndFeatures('abc_table', sequence, ft1, ABC_rowFeatureMap, '#div1', 4)">Residue Clashes</th>
                     <th title="The number of possible atom clashes found in the model - lower is better"
                        onclick="sortTableAndFeatures('abc_table', sequence, ft1, ABC_rowFeatureMap, '#div1', 5)">Atom Clashes</th>
                    <th title="Link to a visualisation of the model and it's corresponding PAE plot">Model visualisations</th>
                </tr>
            </thead>
            <tbody>
                <tr v-for="abcmodel in abc_models" :data-feature-name="abcmodel.model_id">
                    <td><a v-bind:href="abcmodel.model_path" target="_blank">{{ abcmodel.model_id }}</a></td>
                    <td>{{ abcmodel.model_source }}</td>
                    <td>{{ abcmodel.avg_plddt | decimalPlaces }}</td>
                    <td>{{ abcmodel.h_score }}</td>
                    <td>{{ abcmodel.residue_clashes }}</td>
                    <td>{{ abcmodel.atom_clashes }}</td>
                    <td><a v-bind:href="abcmodel.pae_path" target="_blank"><button :class="getButtonClass(abcmodel.model_source)">Click for PAE Plot</button></a></td>
                </tr>
            </tbody>
        </table>
    </div>
    `
});

Vue.component('abc-feature-viewer', {
    data: function () {
        if (typeof window.sequence === 'undefined') {
            window.sequence = this.$root.sequence;
        }
        return {
            abc_models: this.$root.abc_models,
            chain_data: this.$root.chain_data,
            abc_features: [],
        }
    },
    template: `
      <div id="abc-feature-viewer-container" class="content"></div>
    `,
    methods: {
        addFeature(feature) {
            this.abc_features.push(feature);
            ft1.addFeature(feature);
            var rowId = feature.filter;
            window.ABC_rowFeatureMap[rowId] = feature;
        },
        generateABCFeatures() {
            const colors = {
                'v_low': '#FF7D45',
                'low': '#FFDB13',
                'confident': '#65CBF3',
                'v_high': '#0053D6'
            };

            const descriptions = {
                'v_low': 'Very Low Confidence (pLDDT < 50)',
                'low': 'Low Confidence (70 > pLDDT > 50)',
                'confident': 'Confident (90 > pLDDT > 70)',
                'v_high': 'Very High Confidence (pLDDT > 90)'
            };

            const chain_colours = [
                '#991999', // PyMol deeppurple (0.6, 0.1, 0.6)
                '#00BFBF', // PyMol teal (0, 0.75, 0.75)
                '#e9967a', // salmon
                '#009e73',
                '#f0e442',
                '#0072b2',
                '#d55e00',
                '#cc79a7'
            ];

            if (this.chain_data) {
                chain_data = [];
                let colorIndex = 0;
                const colorsLength = chain_colours.length;

                for (const [chain, [start, end]] of Object.entries(this.chain_data)) {
                    chain_data.push({
                        x: start,
                        y: end,
                        color: chain_colours[colorIndex],
                        description: chain
                    });
                    colorIndex = (colorIndex + 1) % colorsLength;
                }
            }

            let chain_feature = {
                data: chain_data,
                name: 'Chain Information',
                className: 'chains',
                type: "multipleRect",
                filter: 'chains',
            };
            this.addFeature(chain_feature);

            if (this.abc_models) {
                this.abc_models.forEach(model => {
                    const modelName = model.model_id;

                    let data = [];
                    for (const [confidence, regions] of Object.entries(model.plddt_regions)) {
                        regions.forEach(([start, end]) => {
                            data.push({
                                x: start,
                                y: end,
                                color: colors[confidence],
                                description: descriptions[confidence],
                            });
                        });
                    }
                    let feature = {
                        data: data,
                        name: modelName,
                        className: modelName,
                        type: "multipleRect",
                        filter: modelName,
                    };
                    this.addFeature(feature);
                });
            }
        }

    },
    mounted() {
        window.ABC_rowFeatureMap = {};
        var options = {
            showAxis: true,
            showSequence: true,
            brushActive: true,
            toolbar:true,
            bubbleHelp:true,
            zoomMax:10,
        };
        window.ft1 = new FeatureViewer.createFeature(sequence,"#div1", options);
        this.generateABCFeatures();

    selectTableAndFeatures(ft1);
    collapseDiv("collapsible1");
    }
});

new Vue({
    el: '#app',
    data: {
        abc_models: abc_data.models,
        sequence: abc_data.sequence,
        chain_data: abc_data.chain_data,
        plotly_path: abc_data.plotly_path,
    },
})
