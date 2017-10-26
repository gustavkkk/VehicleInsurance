
var uploader;

var API_ENDPOINT = 'https://api.openalpr.com';
//var API_ENDPOINT = 'http://localhost:8888'

usStates = {
    AL: "Alabama",AK: "Alaska",AS: "American Samoa",AZ: "Arizona",AR: "Arkansas",CA: "California",CO: "Colorado",CT: "Connecticut",
    DE: "Delaware",DC: "District of Columbia",FL: "Florida",GA: "Georgia",HI: "Hawaii",
    ID: "Idaho",IL: "Illinois",IN: "Indiana",IA: "Iowa",KS: "Kansas",KY: "Kentucky",LA: "Louisiana",ME: "Maine",
    MD: "Maryland",MA: "Massachusetts",MI: "Michigan",MN: "Minnesota",MS: "Mississippi",MO: "Missouri",MT: "Montana",NE: "Nebraska",
    NV: "Nevada",NH: "New Hampshire",NJ: "New Jersey",NM: "New Mexico",NY: "New York",NC: "North Carolina",ND: "North Dakota",
    OH: "Ohio",OK: "Oklahoma",OR: "Oregon",PA: "Pennsylvania",PR: "Puerto Rico",RI: "Rhode Island",
    SC: "South Carolina",SD: "South Dakota",TN: "Tennessee",TX: "Texas",UT: "Utah",VT: "Vermont",VA: "Virginia",
    WA: "Washington",WV: "West Virginia",WI: "Wisconsin",WY: "Wyoming"
    }

function reset_demo_image()
{
    $("#cloud_api_demo .demo_preview_area").removeClass('hide');
    $("#cloud_api_demo .demo_initial_overlay").hide();

    $("#cloud_api_demo .error_text").clearQueue().hide();
    $("#cloud_api_demo .demo_preview_img").empty();
    $("#cloud_api_demo .demo_preview_info").empty();
    $("#cloud_api_demo").addClass("loading");
    $("#cloud_api_demo .demo_preview_img_container").animate({height: 500}, 200);
}

function two_decimals(val)
{
    return Math.round(val * 100) / 100;
}
function fill_demo_info(response)
{
    $info_area = $("#cloud_api_demo .demo_preview_info");




    if (response.results !== undefined)
    {
        for (var i = 0; i < response.results.length; i++)
        {
            var plate = response.results[i];

            var popover_html = "<div class='demo_popover_body'><table class='table table-condensed'>";
            for (var z = 0; z < plate.candidates.length; z++)
            {
                var candidate = plate.candidates[z];

                var matches_template_class = '';
                if (candidate.matches_template != 0)
                    matches_template_class = 'info';

                popover_html += "<tr class='" + matches_template_class + "'><td>" + candidate.plate + "</td><td>" + two_decimals(candidate.confidence) + "%</td></tr>";
            }
            popover_html += '</div></table>';

            $candidate_popover = $('<a href="#" class="demo_info_candidate_popover" onClick="return false;" title="License Plate Top N" data-toggle="popover" data-trigger="focus" tabindex="-1" data-placement="right" data-content="' + popover_html + '"><span class="glyphicon glyphicon-sort-by-attributes-alt glyph-clickable"></span></a>');
            $candidate_popover.popover({ html : true});
            $info_area.append($candidate_popover);

            $info_area.append('<div class="demo_info_header">License Plate</div><div class="demo_info_value">' + plate.plate + ' - ' + two_decimals(plate.confidence) + '%</div>');

            if (plate.region.length > 0 != "" && plate.region_confidence > 0)
            {
                var plate_name = usStates[plate.region.toUpperCase()];
                if (plate_name === undefined)
                    plate_name = plate.region;
                $info_area.append('<div class="demo_info_value">' + plate_name + ' - ' + two_decimals(plate.region_confidence) + '%</div>');
            }



            var classifier_list = [];

            if (plate.vehicle.color !== undefined && plate.vehicle.color.length > 0)
            {
                classifier_list.push('Vehicle Color');
                classifier_list.push(plate.vehicle.color);
            }

            if (plate.vehicle.make !== undefined && plate.vehicle.make.length > 0)
            {
                classifier_list.push('Vehicle Make');
                classifier_list.push(plate.vehicle.make);
            }

            if (plate.vehicle.make_model !== undefined && plate.vehicle.make_model.length > 0)
            {
                classifier_list.push('Vehicle Make-Model');
                classifier_list.push(plate.vehicle.make_model);
            }

            if (plate.vehicle.body_type !== undefined && plate.vehicle.body_type.length > 0)
            {
                classifier_list.push('Vehicle Type');
                classifier_list.push(plate.vehicle.body_type);
            }

            for (var classifier_idx = 0; classifier_idx < classifier_list.length; classifier_idx += 2)
            {
                classifier_name = classifier_list[classifier_idx]
                classifier_data = classifier_list[classifier_idx+1]

                if (classifier_data !== undefined && classifier_data.length > 0)
                {
                    var popover_html = "<div class='demo_popover_body'><table class='table table-condensed'>";

                    for (var idx = 0; idx < classifier_data.length; idx++)
                    {
                        candidate = classifier_data[idx];
                        popover_html += "<tr><td>" + candidate.name + "</td><td>" + two_decimals(candidate.confidence) + "%</td></tr>";
                        //popover_html += "<div class='candidate_info'>" + candidate.name + " - " + two_decimals(candidate.confidence) + "%</div>";
                    }
                    popover_html += '</div></table>';
                    $candidate_popover = $('<a href="#" class="demo_info_candidate_popover" onClick="return false;" title="' + classifier_name + ' Top N" data-toggle="popover" data-trigger="focus" tabindex="-1" data-placement="right" data-content="' + popover_html + '"><span class="glyphicon glyphicon-sort-by-attributes-alt glyph-clickable"></span></a>');
                    $candidate_popover.popover({ html : true});
                    $info_area.append($candidate_popover);

                    $info_area.append('<div class="demo_info_header">' + classifier_name + '</div><div class="demo_info_value">' + classifier_data[0].name + ' - ' + two_decimals(classifier_data[0].confidence) + '%</div>');
                }
            }

            $info_area.append('<hr />');

        }
    }





    var rounded_processing_time = Math.round(response.processing_time.total) / 1000;

    $info_area.append('<div class="demo_info_header">Total Processing Time:</div><div class="demo_info_value">' + rounded_processing_time + ' s</div>');

}

function load_demo_image(response)
{
    $img_canvas_container = $("#cloud_api_demo .demo_preview_img")

    $("#cloud_api_demo").removeClass("loading");

    var MAX_WIDTH = $img_canvas_container.outerWidth();
    var MAX_HEIGHT = $img_canvas_container.outerHeight();

    $img_canvas_container.append('<div id="demo_preview_canvas"></div>');

    var paper = Raphael("demo_preview_canvas", response.img_width, response.img_height );

    var scale_factor = 1.0;
    if (response.img_width > MAX_WIDTH)
        scale_factor = MAX_WIDTH / response.img_width;

    new_height = response.img_height * scale_factor;
    if (new_height > MAX_HEIGHT)
        scale_factor = MAX_HEIGHT / response.img_height;

    var scaled_width = response.img_width * scale_factor;
    var scaled_height = response.img_height * scale_factor;

    $("#cloud_api_demo .demo_preview_img_container").animate({height: scaled_height + 20}, 200);

    paper.setSize(  scaled_width, scaled_height );

    paper.image(response.image_bytes_prefix + response.image_bytes, 0, 0, scaled_width, scaled_height);

    if (response.results !== undefined)
    {
        for (var i = 0; i < response.results.length; i++)
        {
            var plate = response.results[i];
            var vertex = 0;
            for (var vertex = 0; vertex < 4; vertex++)
            {
                var x1 = plate.coordinates[vertex%4].x * scale_factor;
                var y1 = plate.coordinates[vertex%4].y * scale_factor;
                var x2 = plate.coordinates[(vertex+1)%4].x * scale_factor;
                var y2 = plate.coordinates[(vertex+1)%4].y * scale_factor;
                var stroke = paper.path("M" + x1 + " " + y1 + "L" + x2 + " " + y2);
                stroke.attr ("stroke-width", "4");
                stroke.attr ("stroke", "#FF00FF");
            }

        }
    }

    fill_demo_info(response);
}


function show_error(errortext)
{
    $("#cloud_api_demo .error_text").text(errortext).clearQueue().fadeIn(250).delay(5000).fadeOut(250);
    $("#cloud_api_demo").removeClass("loading");
}

function build_search_url()
{


    var countries = '';
    $('#cloud_api_demo .countries option:selected').each(function() {
        countries = countries + $(this).val() + ',';
    });
    // Trim off last comma, if it exists
    if (countries.indexOf(',', this.length - ','.length) !== -1)
        countries = countries.substring(0, countries.length - 1);

    if ($('#cloud_api_demo .upload_type option:selected').val() == "byurl")
    {
        new_url = API_ENDPOINT + '/v2/recognize_url?recognize_vehicle=1&country=' + countries + '&secret_key=' + cloudapi_secret_key + '&return_image=true';
    }
    else
        new_url = API_ENDPOINT + '/v2/recognize?recognize_vehicle=1&country=' + countries + '&secret_key=' + cloudapi_secret_key + '&return_image=true';
    uploader.setOptions({'url': new_url});

    return new_url;
}

function clean_multiline_template(f) {

    return f.toString().
      replace(/^[^\/]+\/\*!?/, '').
      replace(/\*\/[^\/]+$/, '');
}

var base_demo_template = clean_multiline_template(function() {/*!
    <div class="loading_overlay">
      <div class="sk-cube-grid">
        <div class="sk-cube sk-cube1"></div>
        <div class="sk-cube sk-cube2"></div>
        <div class="sk-cube sk-cube3"></div>
        <div class="sk-cube sk-cube4"></div>
        <div class="sk-cube sk-cube5"></div>
        <div class="sk-cube sk-cube6"></div>
        <div class="sk-cube sk-cube7"></div>
        <div class="sk-cube sk-cube8"></div>
        <div class="sk-cube sk-cube9"></div>
      </div>
       <!-- <svg width="110px" height="110px" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid" class="uil-gears"><rect x="0" y="0" width="100" height="100" fill="none" class="bk"></rect><g transform="translate(-20,-20)"><path d="M79.9,52.6C80,51.8,80,50.9,80,50s0-1.8-0.1-2.6l-5.1-0.4c-0.3-2.4-0.9-4.6-1.8-6.7l4.2-2.9c-0.7-1.6-1.6-3.1-2.6-4.5 L70,35c-1.4-1.9-3.1-3.5-4.9-4.9l2.2-4.6c-1.4-1-2.9-1.9-4.5-2.6L59.8,27c-2.1-0.9-4.4-1.5-6.7-1.8l-0.4-5.1C51.8,20,50.9,20,50,20 s-1.8,0-2.6,0.1l-0.4,5.1c-2.4,0.3-4.6,0.9-6.7,1.8l-2.9-4.1c-1.6,0.7-3.1,1.6-4.5,2.6l2.1,4.6c-1.9,1.4-3.5,3.1-5,4.9l-4.5-2.1 c-1,1.4-1.9,2.9-2.6,4.5l4.1,2.9c-0.9,2.1-1.5,4.4-1.8,6.8l-5,0.4C20,48.2,20,49.1,20,50s0,1.8,0.1,2.6l5,0.4 c0.3,2.4,0.9,4.7,1.8,6.8l-4.1,2.9c0.7,1.6,1.6,3.1,2.6,4.5l4.5-2.1c1.4,1.9,3.1,3.5,5,4.9l-2.1,4.6c1.4,1,2.9,1.9,4.5,2.6l2.9-4.1 c2.1,0.9,4.4,1.5,6.7,1.8l0.4,5.1C48.2,80,49.1,80,50,80s1.8,0,2.6-0.1l0.4-5.1c2.3-0.3,4.6-0.9,6.7-1.8l2.9,4.2 c1.6-0.7,3.1-1.6,4.5-2.6L65,69.9c1.9-1.4,3.5-3,4.9-4.9l4.6,2.2c1-1.4,1.9-2.9,2.6-4.5L73,59.8c0.9-2.1,1.5-4.4,1.8-6.7L79.9,52.6 z M50,65c-8.3,0-15-6.7-15-15c0-8.3,6.7-15,15-15s15,6.7,15,15C65,58.3,58.3,65,50,65z" fill="#8f7f59" transform="rotate(15 50 50)"><animateTransform attributeName="transform" type="rotate" from="90 50 50" to="0 50 50" dur="1s" repeatCount="indefinite"></animateTransform></path></g><g transform="translate(20,20) rotate(15 50 50)"><path d="M79.9,52.6C80,51.8,80,50.9,80,50s0-1.8-0.1-2.6l-5.1-0.4c-0.3-2.4-0.9-4.6-1.8-6.7l4.2-2.9c-0.7-1.6-1.6-3.1-2.6-4.5 L70,35c-1.4-1.9-3.1-3.5-4.9-4.9l2.2-4.6c-1.4-1-2.9-1.9-4.5-2.6L59.8,27c-2.1-0.9-4.4-1.5-6.7-1.8l-0.4-5.1C51.8,20,50.9,20,50,20 s-1.8,0-2.6,0.1l-0.4,5.1c-2.4,0.3-4.6,0.9-6.7,1.8l-2.9-4.1c-1.6,0.7-3.1,1.6-4.5,2.6l2.1,4.6c-1.9,1.4-3.5,3.1-5,4.9l-4.5-2.1 c-1,1.4-1.9,2.9-2.6,4.5l4.1,2.9c-0.9,2.1-1.5,4.4-1.8,6.8l-5,0.4C20,48.2,20,49.1,20,50s0,1.8,0.1,2.6l5,0.4 c0.3,2.4,0.9,4.7,1.8,6.8l-4.1,2.9c0.7,1.6,1.6,3.1,2.6,4.5l4.5-2.1c1.4,1.9,3.1,3.5,5,4.9l-2.1,4.6c1.4,1,2.9,1.9,4.5,2.6l2.9-4.1 c2.1,0.9,4.4,1.5,6.7,1.8l0.4,5.1C48.2,80,49.1,80,50,80s1.8,0,2.6-0.1l0.4-5.1c2.3-0.3,4.6-0.9,6.7-1.8l2.9,4.2 c1.6-0.7,3.1-1.6,4.5-2.6L65,69.9c1.9-1.4,3.5-3,4.9-4.9l4.6,2.2c1-1.4,1.9-2.9,2.6-4.5L73,59.8c0.9-2.1,1.5-4.4,1.8-6.7L79.9,52.6 z M50,65c-8.3,0-15-6.7-15-15c0-8.3,6.7-15,15-15s15,6.7,15,15C65,58.3,58.3,65,50,65z" fill="#9f9fab" transform="rotate(75 50 50)"><animateTransform attributeName="transform" type="rotate" from="0 50 50" to="90 50 50" dur="1s" repeatCount="indefinite"></animateTransform></path></g></svg> -->
    </div>

    <div class="demo_preview_img_container img-rounded">

        <div class="demo_initial_overlay img-rounded">
            Drag and Drop an Image Here<br />
            Or use the form below to upload.
        </div>

            <div class="demo_preview_area hide demo_preview_info"></div>
            <div class="demo_preview_area hide demo_preview_img"></div>
    </div>



    <div class="demo_form_controls">


        <div class="form-inline">
            <span class="upload_form_label">Plate Style: </span>
            <select class="countries form-control" >
            </select>
        </div>

        <div class="form-inline upload_section byfile">
            <span class="upload_form_label">Upload: </span>

            <select class="form-control upload_type">
                <option value="byfile">File</option>
                <option value="byurl">Url</option>
            </select>

            <input type="text" class="form-control url_entry" placeholder="Image URL" />

            <div class="demo_submit_buttons">
                <button class="btn upload_url_button">Analyze URL</button>
                <button class="btn upload_file_button">Upload Image</button>
            </div>
        </div>
    </div>

    <div class="error_text"></div>
*/});

$(document).ready(function() {

    // On page load, load the country selection (from AJAX request)
    $.ajax({
        type: 'GET',
        url: API_ENDPOINT + '/v2/config'
    })
    .done(function( data ) {
        $selector = $('#cloud_api_demo .countries').first();
        for (i in data.countries) {
            // Special case for US, it's enabled by default
            var code = data.countries[i].code;
            var country_name = data.countries[i].name;

            if (code == 'us' || code == 'na')
                $selector.append( $('<option selected></option>').val(code).html(country_name) );
            else
                $selector.append( $('<option></option>').val(code).html(country_name) );

        }

        $('#cloud_api_demo .countries').multiselect({

            onChange: function(element, checked) {
                build_search_url();
            }
        })
        .error(function( jqXHR, textStatus, errorThrown  ) {
            show_error("Error loading country list, please refresh the page");
      });


      build_search_url();
    });

    $("#cloud_api_demo").html($(base_demo_template));

    var MaxSizeKb = 4000;
    valid_extensions = ["jpg", "jpeg", "png", "gif"];
    uploader = new ss.SimpleUpload({
          button: $("#cloud_api_demo button.upload_file_button"), // HTML element used as upload button
          url: "", // URL of server-side upload handler
          dropzone: $("#cloud_api_demo .demo_preview_img_container"),
          cors: true,
          //debug: true,
          responseType: "json",
          maxSize: MaxSizeKb, // kilobytes
          allowedExtensions: valid_extensions,
          name: 'image',
          onSubmit: function(filename, extension, uploadBtn, filesize) {
             reset_demo_image();

          },
          onComplete: function(filename, response, uploadBtn, filesize) {

            load_demo_image(response)

          },
          onSizeError: function(filename, fileSize)
          {
            show_error("File too large.  Maximum upload size for the demo is " + MaxSizeKb + " kB");
          },
          onExtError: function(filename, extension)
          {
            show_error("Incorrect image type type.  Allowed image extensions: " + valid_extensions.join(", "));
          },
          onError: function(filename, errorType, status, statusText, response, uploadBtn, filesize) {
            show_error(statusText);
          }
    });

    $("#cloud_api_demo .upload_url_button").click(function() {
        url = build_search_url();

         reset_demo_image();

        $.ajax({
          type: 'POST',
          url: url,
          data: {'image_url': $("#cloud_api_demo input.url_entry").val()}
        })
          .done(function( data ) {
            load_demo_image(data);
          })
          .error(function( jqXHR, textStatus, errorThrown  ) {
            if (jqXHR.responseText == undefined ||jqXHR.responseText.length == 0 )
                show_error("Error sending URL");
            else
            {
                error_json = JSON.parse(jqXHR.responseText);
                show_error(error_json['error']);
            }
          });

    });





    $('#cloud_api_demo .upload_type').change(function() {
        upload_type = $(this).val();

        if (upload_type == "byurl")
        {
            $("#cloud_api_demo .upload_section").addClass('byurl');
            $("#cloud_api_demo .upload_section").removeClass('byfile');
        }
        else if (upload_type == "byfile")
        {
            $("#cloud_api_demo .upload_section").removeClass('byurl');
            $("#cloud_api_demo .upload_section").addClass('byfile');
        }

        build_search_url();
    });



});