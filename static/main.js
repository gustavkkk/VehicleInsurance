
$(document).ready(function() {
		/*
     * Replace all SVG images with inline SVG
     */
        jQuery('img.svg').each(function(){
            var $img = jQuery(this);
            var imgID = $img.attr('id');
            var imgClass = $img.attr('class');
            var imgURL = $img.attr('src');

            jQuery.get(imgURL, function(data) {
                // Get the SVG tag, ignore the rest
                var $svg = jQuery(data).find('svg');

                // Add replaced image's ID to the new SVG
                if(typeof imgID !== 'undefined') {
                    $svg = $svg.attr('id', imgID);
                }
                // Add replaced image's classes to the new SVG
                if(typeof imgClass !== 'undefined') {
                    $svg = $svg.attr('class', imgClass+' replaced-svg');
                }

                // Remove any invalid XML tags as per http://validator.w3.org
                $svg = $svg.removeAttr('xmlns:a');

                // Replace image with new SVG
                $img.replaceWith($svg);

            }, 'xml');

        });

    /* ======= Twitter Bootstrap hover dropdown ======= */   
    /* Ref: https://github.com/CWSpear/bootstrap-hover-dropdown */ 
    /* apply dropdownHover to all elements with the data-hover="dropdown" attribute */
    
    $('[data-hover="dropdown"]').dropdownHover();
    
    /* ======= Fixed header when scrolled ======= */    
    $(window).bind('scroll', function() {
         
         if ($(window).scrollTop() > 30) {
             $('#header').addClass('navbar-scrolled');
         }
         else {
             $('#header').removeClass('navbar-scrolled');
             
         }
    });
    
    /* ======= jQuery Placeholder ======= */
    /* Ref: https://github.com/mathiasbynens/jquery-placeholder */
    
    $('input, textarea').placeholder();    
    
    /* ======= jQuery FitVids - Responsive Video ======= */
    /* Ref: https://github.com/davatron5000/FitVids.js/blob/master/README.md */
    
    $(".cloud-video").fitVids();
    
    /* ======= FAQ accordion ======= */
    function toggleIcon(e) {
    $(e.target)
        .prev('.panel-heading')
        .find('.panel-title a')
        .toggleClass('active')
        .find("i.fa")
        .toggleClass('fa-plus-square fa-minus-square');
    }
    $('.panel').on('hidden.bs.collapse', toggleIcon);
    $('.panel').on('shown.bs.collapse', toggleIcon);    
    
    
    /* ======= Header Background Slideshow - Flexslider ======= */    
    /* Ref: https://github.com/woothemes/FlexSlider/wiki/FlexSlider-Properties */
    
    var marketing_slogans = [
    {big: "Open, Accurate, and Affordable", little: "OpenALPR’s Cloud Stream service is now free. <br />Sign up now and start monitoring license plates!"},
    {big: "Essential Security for Property Managers, HOAs and Corporate Campuses", little: "know who is on your property at all times. Offer your tenants better security and peace of mind."},
    {big: "Open for OEM", little: "Enabling technology for hardware and software developers. Cloud API service provides plate, state, color, make and model recognition via a REST API"},
    {big: "No VMS Software or Server Required", little: "All you need is the OpenALPR app for Axis cameras to enable ALPR as a Service."}
    ];
    var marketing_slogan_index = 0;
    $('.bg-slider').flexslider({
        animation: "fade",
        animationSpeed: 600,
        directionNav: false, //remove the default direction-nav - https://github.com/woothemes/FlexSlider/wiki/FlexSlider-Properties
        controlNav: false, //remove the default control-nav
        slideshowSpeed: 8000,
        before: function() {
            marketing_slogan_index = (marketing_slogan_index + 1) % marketing_slogans.length;

            $("section.promo.section h2.title").fadeOut(300, function() {
                $(this).html(marketing_slogans[marketing_slogan_index].big);
            }).fadeIn(300);
            $("section.promo.section p.intro").fadeOut(300, function() {
                $(this).html(marketing_slogans[marketing_slogan_index].little);
            }).fadeIn(300);
        }
    });
	
	/* ======= Stop Video Playing When Close the Modal Window ====== */
    $("#modal-video .close").on("click", function() {
        $("#modal-video iframe").attr("src", $("#modal-video iframe").attr("src"));        
    });
     
    
     /* ======= Testimonial Bootstrap Carousel ======= */
     /* Ref: http://getbootstrap.com/javascript/#carousel */
    $('#testimonials-carousel').carousel({
      interval: 8000 
    });
	
	/* ======= Humburger + menu ======= */
	$('.navbar-toggle').on('click', function() {
			$('#header').addClass('header_bg');
			$('.hamburger').addClass('is-active');
			if($('#navbar-collapse').hasClass('in')){
				 	$('#header').removeClass('header_bg');
					$('.hamburger').removeClass('is-active');
				 }
		});
	$(window).resize(function(){
		if ($(window).width() > 768){
			$('#header').removeClass('header_bg');
			$('#navbar-collapse').removeClass('in');
			$('.hamburger').removeClass('is-active'); 
		}	
	});
		
	//mackboock carousel
		$(".alpr-carousel").owlCarousel({
			items: 1,
			nav: true,
			navText: ["<i class='fa fa-chevron-left'></i>","<i class='fa fa-chevron-right'></i>"],
			loop: true,
			autoplay:true,
		}); 
	//photos carousel
		$(".photos__carousel").owlCarousel({
			items: 3,
			nav: true,
			dots: false,
			slideBy: 2,
			autoWidth:true,
			center:true, 
			loop: true,
			autoplay:true,
			navText: ["<i class='fa fa-chevron-left'></i>","<i class='fa fa-chevron-right'></i>"],
		});
	 
	//fancybox images
	$("[data-fancybox]").fancybox({ 
			// Options will go here
	});
	//fancybox video
	$('[data-fancybox="video"]').fancybox({
		afterLoad : function( instance, current ) {
		// Remove scrollbars and change background
			current.$content.css({
			overflow   : 'visible',
			background : '#000'
		}); 
		},
  		onUpdate : function( instance, current ) {
	    var width,
	        height,
	        ratio = 16 / 9,
	        video = current.$content; 
	    if ( video ) {
	      video.hide();
	      width  = current.$slide.width();
	      height = current.$slide.height() - 100;
	      if ( height * ratio > width ) {
	        height = width / ratio;
	      } else {
	        width = height * ratio;
	      }
	      video.css({
	        width  : width,
	        height : height
	      }).show();
	    }
	  }
	}); 
});
