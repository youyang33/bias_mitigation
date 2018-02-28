(function() {
    var bugout = new debugout(); 
    bugout.autoTrim = false; 
    analysis = {};
    var dataCSV = 'data/bball_v3-25-top.csv';
    var intLogJSON = 'pilot_data/processed/pilot02_int_task1_processed.json';
    var dataSet; 
    var interactionLogs; 
    
    analysis.init = function() {

        $('#goButton').click(function(ev) {
            analysis.runAnalysis(params.width, params.height);
        });

        $('#downloadButton').click(function(ev) {
            var biasFileName = $('#downloadFileName').val();
            bugout.logFilename = biasFileName;
            console.log('Downloading bias logs to ' + biasFileName);

            var biasLogs = ial.getBiasLogs(); 
            bugout.log('------------------------------------------------');
            bugout.log('------------ BIAS LOGS (' + biasLogs.length + ') ------------'); 
            bugout.log('------------------------------------------------');
            for (var log in biasLogs) bugout.log(biasLogs[log]);
            bugout.log();

            bugout.downloadLog();  
            bugout.clear(); 

            ial.printBiasLogs();
        });
        analysis.initializeIAL(); 
    };

    // load in the data and log files
    analysis.initializeIAL = function() {
        d3.queue()
            .defer(d3.csv, dataCSV)
            .defer(d3.json, intLogJSON)
            .awaitAll(function(error, results) {
                if (error) throw error; 
                dataSet = results[0];
                interactionLogs = results[1];
                console.log('init dataset', dataSet);
                console.log('init interaction logs', interactionLogs);
               
                ial.init(dataSet);

                for (var index in interactionLogs) {
                    var dataId = interactionLogs[index]['dataItem']['ial']['id']; 
                    var dataName = interactionLogs[index]['dataItem']['Player']; 
                    //console.log('id = ' + dataId + ', name = ' + dataName); 
                    var eventName = interactionLogs[index]['eventName'];
                    if (eventName.indexOf('AttributeWeight') == -1) 
                    	ial.interactionEnqueue(interactionLogs[index]);
                    else
                    	ial.attributeWeightVectorEnqueue(interactionLogs[index]);
                    
                    var biasResult = ial.computeBias(); // TODO: Compute here after each interaction?
                    //console.log('bias result', biasResult); 
                }
                
                console.log('ial interaction logs', ial.getInteractionQueue());
                console.log('ial attribute weight logs', ial.getAttributeWeightVectorQueue());
            });
    };

    // analyze interaction logs
    analysis.runAnalysis = function(width, height) {
        var biasResult = ial.computeBias(); 
        console.log('bias result', biasResult);
        $('#output_panel').html('');
        $('#output_panel').append('<b>All Metrics:</b> ' + JSON.stringify(biasResult));
    };

})();