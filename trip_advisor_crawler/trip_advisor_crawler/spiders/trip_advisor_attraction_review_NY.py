# -*- coding: utf-8 -*-
import scrapy


class TripAdvisorAttractionsNySpider(scrapy.Spider):
    name = 'trip_advisor_attractions_NY'
    allowed_domains = ['www.tripadvisor.com']
    #start_urls = ['http://www.tripadvisor.com/Attractions-g60763-Activities-a_allAttractions.true-New_York_City_New_York.html/']
    #start_urls = ['https://www.tripadvisor.com/Attractions-g55197-Activities-a_allAttractions.true-Memphis_Tennessee.html']
    #start_urls = ['https://www.tripadvisor.com/Attractions-g35805-Activities-a_allAttractions.true-Chicago_Illinois.html']
    #start_urls = ['https://www.tripadvisor.com/Attractions-g34438-Activities-a_allAttractions.true-Miami_Florida.html']
    #start_urls = ['https://www.tripadvisor.com/Attractions-g60750-Activities-a_allAttractions.true-San_Diego_California.html']
    start_urls = ['https://www.tripadvisor.com/Attractions-g28970-Activities-a_allAttractions.true-Washington_DC_District_of_Columbia.html']

    def parse(self, response):
        # urls have already been dealt with
        urls_set = set()

        # The URL for review
        links = response.xpath('//div[@class="tracking_attraction_title listing_title "]/a/@href').extract()

        # FOREACH reviews link/attraction in this page
        for href in links:
            # Generate absolute URL
            url = response.urljoin(href)

            if url not in urls_set:
                urls_set.add(url)

                yield scrapy.Request(url, callback=self.parse_reviews_summary)

        next_page = response.xpath('//link[@rel="next"]/@href').extract()
        offset_num = int(response.xpath('//div[@class="unified pagination "]/a[contains(@class, "nav next")]/@data-offset').extract_first())

        # return 630
        if next_page and offset_num <= 600:
            url = response.urljoin(next_page[-1])

            yield scrapy.Request(url, callback=self.parse)

    def parse_reviews_summary(self, response):
        links = response.xpath('//div[contains(@class, "location-review-review-list-parts-ReviewTitle__reviewTitle")]/a/@href').extract()

        #reviews = response.xpath('//q[contains(@class, "location-review-review-list-parts-ExpandableReview__reviewText")]/span/text()').extract()

        for href in links:
            url = response.urljoin(href)

            yield scrapy.Request(url, callback=self.parse_review)

        #next_page = response.xpath('//div[contains(@class, "ui_pagination")]/a[@class="ui_button nav next primary "]/@href').extract()
        next_page = response.xpath('//link[@rel="next"]/@href').extract()
        cur_page = int(response.xpath('//span[@class="pageNum current disabled"]/text()').extract_first())

        if next_page and cur_page <= 20:
            # meta: Otherwise redirect 301
            url = response.urljoin(next_page[-1])

            yield scrapy.Request(url, callback=self.parse_reviews_summary)

    def parse_review(self, response):
        usr_code = response.xpath('//div[@class="info_text"]/div/text()').extract_first()
        place = response.xpath('//div[@class="altHeadInline"]/a/text()').extract_first()
        title = response.xpath('//h1[@id="HEADING"]/text()').extract_first()
        ### Rating ui_bubble_rating bubble_30 / 50 => split bubble_30 => split 30
        rate = response.xpath('//div[@class="reviewSelector"]/div/div/span/@class').extract_first().split()[1].split("_")[1]
        review = response.xpath('//p[@class="partial_entry"]/span[@class="fullText "]/text()').extract()[-1]
        month = response.xpath('//div[contains(@class, "reviews_stay_date")]/text()').extract_first().strip()

        yield {
            'usr_code': usr_code.encode("utf-8"),
            'place': place.encode("utf-8"),
            'title': title.encode("utf-8"),
            'rating': int(rate),
            'review': review.encode("utf-8"),
            'travel_date': month
        }
