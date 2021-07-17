# MSc Project - Birkbeck University of London
# Title: Automated Model Selection for Time Series Forecasting
# Author: Jose Manuel Magana Arias
# Version: 0.1 - Summer 2020

# 00 Setup ----

# 00.01 Load libraries ----
#rm(list=ls())
# time series database
library(tsdl)

# summary statistics
library(skimr)
library(formattable)

# data preparation
library(recipes)
library(rsample)
library(tidyverse)
library(tidyquant)

# time series methods
library(timetk)
library(tibbletime)
library(fable)

# accuracy metrics
library(yardstick)

# data visualization
library(ggfortify)
library(scales)

# user interface
library(plotly)
library(shiny)
library(shinythemes)
library(shinyWidgets)
library(shinycssloaders)


# 00.02 Load R functions ----
source(file = "scripts/analysis_functions.R")

# 00.03 Load Python scripts ----
library(reticulate)
use_condaenv("msc_project", required = TRUE)
reticulate::source_python("scripts/lstm.py")

# 00.04 Global variables ----
ts_list_tbl    <- get_ts_list()
method_options <- c("ARIMA",
                    "Neural Network Autoregression",
                    "Long Short-Term Memory Network")

default_ts <- "U.S. annual coffee consumption. 1910 – 1970."
#default_ts   <- "Daily maximum temperatures in Melbourne, Australia, 1981-1990"

# ARIMA Non-seasonal part choices
ar_choices   <- c(0:12)
diff_choices <- c(0, 1)
ma_choices   <- c(0:12)

# ARIMA Seasonal part choices
sar_choices    <- c(0:12)
s_diff_choices <- c(0, 1)
sma_choices    <- c(0:12)
s_max          <- 365

# Neural Network Autoregression choices
nnar_p_choices <- c(0:6)
nnar_P_choices <- c(0:6)
nnar_s_max     <- 365
nnar_networks  <- 100
nnar_max_networks <- 1000


# Long Short-Term Memory model choices
lstm_activation_choices <- c("tanh", "elu", "relu", "sigmoid",
                             "softmax", "softplus", "softsign", "selu")
lstm_optimizer_choices  <- c("rmsprop", "adadelta", "adagrad", "adam", "adamax",
                             "ftrl", "nadam", "sgd")
lstm_loss_choices       <- c("mae", "mape", "mse", "msle", "poisson")

lstm_lags       <- 15
lstm_batch_size <- 90
lstm_epochs     <- 300
lstm_neurons    <- 40

lstm_max_lags       <- 99
lstm_max_batch_size <- 1000
lstm_max_epochs     <- 1000
lstm_max_neurons    <- 100

# 01 User Interface ----
ui <- fluidPage(

  theme = shinytheme("cosmo"),

  # 01.00 Navigation panel ----
  navbarPage(
    title = "Birkbeck College - MSc Data Science Project",
    inverse = FALSE,
    collapsible = TRUE,

    # 01.01 Welcome page ----
    shiny::tabPanel(
      title = "Welcome",
      value = "page_1",
      h1(class = "page-header",
         "Welcome to", strong(em("DeepSight")), tags$small("v1.0")),

    fluidRow(
      column(
        width = 6,
        p(class = "lead", "A time series prediction tool"),
        p("This project provides an empirical comparative evaluation of the",
          "performance of different forecasting techniques to the problem of",
          "univariate time series prediction. This application trains and",
          "tests the performance of different statistical parametric methods",
          "and Artificial Neural Networks (ANNs) with the objective to produce a",
          "data-driven recommendation of the best model to apply for the specific",
          "dataset selected."),
          tags$br(),
        p("As part of this application, the following forecasting techniques",
          "have been implemented:"),
        tags$ol(
          tags$li("Autoregressive Integrated Moving Average (ARIMA) model"),
          tags$li("Neural Networks Autoregression (NNARs)"),
          tags$li("Long Short-Term Memory Networks (LSTM)")
        )
      ),
      column(
        width = 6,
        tags$blockquote(
          class = "blockquote-reverse",
          p("- Hey, Doc! Where you goin' now? Back to the future?",
            tags$br(),
          p("- Nope. Already been there"),
          tags$footer(
            tags$cite(
              title = "x", "Dr. Emmett Brown, Marty McFly, Back to the Future")))
        )
      )
    ),
    ),

    # 01.02 sidebarPanel ----
    tabPanel(
      title = "Time Series Forecasting",
      value = "page_2",
      h1(class = "page-header", "Time Series Forecasting Methods"),
      sidebarLayout(
       sidebarPanel(width = 3,
          fluidRow(
            column(width = 12,
              h4("Select a dataset")
            ),
            column(width = 7,
                  conditionalPanel(
                    condition = "!input.is_csv",
                    pickerInput(
                      inputId = "ts_selection",
                      choices = ts_list_tbl,
                      multiple = FALSE,
                      selected = default_ts,
                      options = pickerOptions(
                        actionsBox = FALSE,
                        liveSearch = TRUE,
                        size = 10,
                      )
                    )
                  ),
                  conditionalPanel(
                    condition = "input.is_csv",
                    fileInput(inputId = "load_csv_file",
                              label = NULL,
                              buttonLabel = "Browse",
                              accept = ".csv"
                    )
                  )
            ),
            # 01.02.00 CSV File user interface ----
            column(width = 5,
              switchInput(inputId = "is_csv",
                          label = "csv file",
                          onStatus = "success",
                          size = "small",
                          value = FALSE)
            ),
            column(width = 12,
              conditionalPanel(
                condition = "input.is_csv && output.csv_cols > 1",
                # time / date variable picker
                uiOutput('ui_csv_indep_vars'),
                # dependent variable picker
                uiOutput('ui_csv_dep_vars')
              )
            ),

            column(width = 12,
                   h4("Select forecasting mode"),
                   switchInput(inputId = "is_autoforecast",
                               label = "AutoForecast",
                               onStatus = "success",
                               width = "180px",
                               size = "small",
                               value = TRUE)
            )
          ),
          # 01.02.01 Forecast mode panel ----
          conditionalPanel(
            h4("Select the forecasting methods to run"),
            condition = "input.is_autoforecast",

            prettyCheckbox(
              inputId = "arima_check",
              label = "ARIMA model",
              value = TRUE,
              status = "success",
              outline = TRUE
            ),
            prettyCheckbox(
              inputId = "nnar_check",
              value = TRUE,
              label = "Neural Network Autoregression",
              status = "success",
              outline = TRUE
            ),
            # prettyCheckbox(
            #   inputId = "lstm_check",
            #   label = "Long Short-Term Memory Networks",
            #   status = "success",
            #   outline = TRUE
            # ),

          ),
          conditionalPanel(
            h4("Select a forecasting method"),
            condition = "!input.is_autoforecast",
            selectInput(
              inputId = "method",
              label = NULL,
              choices = method_options,
              selected = NULL,
              multiple = FALSE,
              selectize = FALSE,
              width = NULL,
              size = NULL
            )
          ),
          conditionalPanel(
            # 01.02.02 ARIMA conditional panel ----
            h4("Enter model specification"),
            condition = "!input.is_autoforecast && input.method == 'ARIMA'",
            fluidRow(
              #br(),
              # AR order
              column(width = 6,
                     style = list("padding-right: 5px;"),
                     selectInput("p","Non-seasonal autoregressive",
                                 choices = ar_choices,
                                 selected = "1")
              ),
              # Differencing
              column(width = 6,
                     style = list("padding-left: 5px;"),
                     selectInput("d", "Non-seasonal differences",
                                 choices = diff_choices,
                                 selected = 0)
              ),
              # MA term
              column(width = 6,
                     style = list("padding-right: 5px;"),
                     selectInput("q", "Non-seasonal moving-avg",
                                 choices = ma_choices,
                                 selected = 0)
              ),
              # SAR term
              column(width = 6,
                     style = list("padding-left: 5px;"),
                     selectInput("P", "Seasonal autoregressive",
                                 choices = sar_choices,
                                 selected = 0)
              ),
              # Seasonal difference
              column(width = 6,
                     style = list("padding-right: 5px;"),
                     selectInput("D", "Seasonal differences",
                                 choices = s_diff_choices,
                                 selected = 0)
              ),
              # SMA term
              column(width = 6,
                     style = list("padding-left: 5px;"),
                     selectInput("Q", "Seasonal moving-avg",
                                 choices = sma_choices,
                                 selected = 0)
              ),
              # Seasonality
              column(width = 6,
                     style = list("padding-right: 5px;"),
                     numericInput("s", "Seasonality",
                                  value = 1,
                                  min = 1,
                                  max = s_max,
                                  step = 1)
              ),
            )
          ),
          conditionalPanel(
            # 01.02.03 NNAR conditional panel ----
            h4("Enter model specification"),
            condition = "!input.is_autoforecast && input.method == 'Neural Network Autoregression'",
            fluidRow(
              #br(),
              # AR order
              column(width = 6,
                     style = list("padding-right: 5px;"),
                     selectInput("nnar_p","Non-seasonal autoregressive",
                                 choices = nnar_p_choices,
                                 selected = 1)
              ),
              # SAR order
              column(width = 6,
                     style = list("padding-left: 5px;"),
                     selectInput("nnar_P","Seasonal autoregressive",
                                 choices = nnar_P_choices,
                                 selected = 0)
              ),
              # Seasonality
              column(width = 6,
                     style = list("padding-right: 5px;"),
                     numericInput("nnar_period", "Seasonality",
                                  value = 1,
                                  min = 1,
                                  max = nnar_s_max,
                                  step = 1)
              ),
              # Number of networks to fit with different random starting weights
              # These are then averaged when producing forecasts.
              column(width = 6,
                     style = list("padding-left: 5px;"),
                     numericInput("nnar_n", "Number of networks",
                                  value = nnar_networks,
                                  min = 1,
                                  max = nnar_max_networks,
                                  step = 1)
              ),
              column(width = 6,
                     p("Simulate prediction intervals"),
                     style = list("padding-right: 5px;"),
                     switchInput(inputId = "nnar_pred_intervals",
                                 label = "Intervals",
                                 onStatus = "success",
                                 offStatus = "danger",
                                 size = "small",
                                 value = TRUE)
              )
            )
      ),
      conditionalPanel(
        # 01.02.04 LSTM conditional panel ----
        h4("Enter model specification"),
        condition = "!input.is_autoforecast && input.method == 'Long Short-Term Memory Network'",
        fluidRow(
          #br(),
          # Timesteps (lags)
          column(width = 6,
                 style = list("padding-right: 5px;"),
                 numericInput("lstm_lags", "Timesteps (lags)",
                              value = lstm_lags,
                              min = 1,
                              max = lstm_max_lags,
                              step = 1)
          ),
          # Epochs
          column(width = 6,
                 style = list("padding-left: 5px;"),
                 numericInput("lstm_epochs", "Epochs",
                              value = lstm_epochs,
                              min = 1,
                              max = lstm_max_epochs,
                              step = 1)
          ),
          # Batch size
          column(width = 6,
                 style = list("padding-right: 5px;"),
                 numericInput("lstm_batch_size", "Batch size",
                              value = lstm_batch_size,
                              min = 1,
                              max = lstm_max_batch_size,
                              step = 1)
          ),
          # Number of units in hidden layer
          column(width = 6,
                 style = list("padding-left: 5px;"),
                 numericInput("lstm_neurons", "Neurons",
                              value = lstm_neurons,
                              min = 1,
                              max = lstm_max_neurons,
                              step = 1)
          ),
          # Activation function
          column(width = 6,
                 style = list("padding-right: 5px;"),
                 selectInput(inputId = "lstm_activation_function",
                             label = "Activation function",
                             choices = lstm_activation_choices,
                             selected = "relu"),
          ),
          # Optimization algorithm
          column(width = 6,
                 style = list("padding-left: 5px;"),
                 selectInput(inputId = "lstm_optimizer",
                             label = "Optimization algorithm",
                             choices = lstm_optimizer_choices,
                             selected = "rmsprop"),
          ),
          # Loss function
          column(width = 6,
                 style = list("padding-right: 5px;"),
                 selectInput(inputId = "lstm_loss",
                             label = "Loss function to minimize",
                             choices = lstm_loss_choices,
                             selected = "mae"),
          )
        )
      ),
          sliderInput("train_prop",
                      h4("Select a proportion for the train dataset (%):"),
                      min = 50, max = 90, step = 5, value = 80
          ),
          conditionalPanel(
            condition = "!input.is_autoforecast",
            hr(),
            column(width = 6,
                   actionButton(inputId = "train_button",
                                label = "Train Model",
                                icon = icon("cog")
                   )
            ),
            column(width = 6,
                   actionButton(inputId = "run_forecast_button",
                                label = "Run Forecast",
                                icon = icon("chart-line")
                   )
            ),
          ),
          conditionalPanel(
            condition = "input.is_autoforecast",
            hr(),
            column(width = 6,
                   actionButton(inputId = "auto_forecast_button",
                                label = "Run AutoForecast",
                                icon = icon("chart-line")
                   )
            ),
          ),
        br(),
        br(),
      ),
      mainPanel(
        tabsetPanel(
          id = "manual_model_menu",
          type = "hidden",
          # 01.03 Navigation tabs ----
          tabPanel(title = "manual_model_tabs",
                    tabsetPanel(
                      id = "tab_1",
                      type = "tabs",
                      # 01.03.01 Tab I: Time series ----
                      tabPanel(title = "Time Series",
                              column(
                                width = 12,
                                h3(textOutput(outputId = "ts_name")),
                                plotlyOutput(outputId = "dataviz_plot"),
                                hr(), # Horizontal line
                              ),
                              # Summary statistics
                              column(width = 9,
                                h3("Summary statistics"),
                                br(),
                                h4(textOutput(outputId = "ts_description")),
                                formattableOutput(outputId = "summary_stats"),
                              ),
                      ),
                      # 01.03.02 Tab II: Autocorrelation plots ----
                      tabPanel(title = "Autocorrelation Plots",
                              column(
                                width = 12,
                                h3("Autocorrelation and Partial-Autocorrelation"),
                                plotlyOutput(outputId = "ac_plots")
                              )
                      ),
                      # 01.03.03 Tab III: Model Learning ----
                      tabPanel(title = "Model Learning",
                               column(width = 12,
                                 h3("Train-Test split"),
                                 plotlyOutput(outputId = "ts_split_plot"),
                               ),
                               fluidRow(
                                 column(width = 12,
                                        hr(), # Horizontal line
                                 ),
                                 # Train report
                                 column(width = 6,
                                        conditionalPanel(
                                          h3("Model fit on the training dataset"),
                                          condition = "input.is_autoforecast || (!input.is_autoforecast && input.method != NULL)",
                                          h4(textOutput(outputId = "train_method", inline = TRUE),
                                             textOutput(outputId = "train_formula", inline = TRUE)
                                             ),
                                          # Render report with spinner
                                          withSpinner(
                                            ui_element = verbatimTextOutput(outputId = "model_report"),
                                            type = 8,
                                            hide.ui = FALSE
                                          )
                                        )
                                 ),
                                 column(width = 6,
                                        conditionalPanel(
                                          condition = "input.is_autoforecast || (!input.is_autoforecast && input.method == 'ARIMA')",
                                          h3(textOutput(outputId = "train_detail_header")),
                                          # Render report with spinner
                                          withSpinner(
                                            ui_element = tableOutput(outputId = "train_detail"),
                                            type = 8,
                                            hide.ui = FALSE
                                          )
                                        )
                                 )
                                )
                      ),
                      # 01.03.04 Tab IV: Model Testing ----
                      tabPanel(title = "Model Testing",
                               fluidRow(
                                 column(width = 12,
                                        h3("Performance on test dataset"),
                                        withSpinner(
                                          ui_element = plotOutput(outputId = "forecast_plot"),
                                          type = 8,
                                          hide.ui = FALSE
                                        )
                                 ),
                                 column(width = 12,
                                        hr()
                                 ),
                                 # Forecast accuracy report
                                 column(width = 12,
                                        h3("Forecast (out-of-sample) accuracy"),
                                        h4(textOutput(outputId = "accuracy_report_header")),
                                        withSpinner(
                                          ui_element = tableOutput(outputId = "accuracy_report"),
                                          type = 8,
                                          hide.ui = FALSE
                                        )
                                 )
                               )
                      )
                    )
          )
        )
      )
      )
    ),

    # 01.04 Documentation ----
    navbarMenu(
      title = "Documentation",
      tabPanel(title = "Project Proposal",
               value = "page_4_1",
               h1(class = "page-header", "Project Proposal"),
               p(class = "lead", #"A Shiny app on ",
                 a(href = "https://jmagan01.github.io/msc-project-proposal/",
                   target = "_blank",
                   "Automated model selection for time series forecasting")),
      ),
      tabPanel(title = "Project Report",
               value = "page_4_2",
               h1(class = "page-header", "Project Report"),
               p("Placeholder"),
      ),
      "----",
      tabPanel(title = "References",
               value = "page_4_3",
               h1(class = "page-header", "References"),
               p("Placeholder"),
      )
    )
  ),
  # 01.05 CSS code ----
  tags$head(
    tags$style(".page-header {margin-top: 0px; margin-bottom: 20px}"),
    tags$style(".selectize-input {line-height: 25px;}"),
    tags$style(".selectize-input.dropdown-active {line-height: 25px;}"),
    tags$style("#model_report{background: white; border-style: none;}"),
    tags$style(".control-label {font-size: 12pt; font-weight: normal;}"),
    tags$style(".irs-bar {background: #222222; border-bottom-color: #222222; border-top-color: #222222;}"),
    tags$style(".irs-bar-edge {background: #222222; border: #222222;"),
    tags$style(".irs-from {background-color: #222222;}"),
    tags$style(".irs-single {font-size: 12px; background-color: #222222;}"),
    tags$style(".irs-grid-pol {background-color: #222222;}"),
    tags$style(".irs-grid-text {font-size: 12px;"),

    # Hide any error message
    tags$style(type="text/css",
               ".shiny-output-error { visibility: hidden; }",
               ".shiny-output-error:before { visibility: hidden; }")
  )
)

# 02 Server ----
server <- function(input, output, session){

  # 02.00 ==== LOAD CSV FILE ==== ----

  # 02.00.01 Load CSV file ----
  values <- reactiveValues(csv_data = NULL)

  observeEvent(input$load_csv_file, {

    csv_file <- input$load_csv_file

    if(is_csv_ext()){
      # Load data
      execute_safely(
        values$csv_data <- read.csv(csv_file$datapath, header = TRUE)
      )
      # Display success sweet alert
      sendSweetAlert(
        session = session,
        title = "Completed!",
        text = "Data successfully loaded",
        type = "success"
      )
    } else {
      # Display error sweet alert
      sendSweetAlert(
        session = session,
        title = "Oops, invalid file type!",
        text = "Please load a csv file",
        type = "error"
      )
    }
  })

  # 02.00.02 Check CSV file ----
  is_csv_ext <- reactive({
    check_file_extension(input$load_csv_file)
  })

  output$csv_cols <- reactive({
    dim(values$csv_data)[2]
  })

  is_csv_ok <- reactive({
    dim(values$csv_data)[2] > 1
  })

  # 02.00.03 Render user interface ----
  output$ui_csv_indep_vars <- renderUI({
    selectInput(inputId = 'time_variable',
                label = h4("Select the time or date variable"),
                choices  = names(values$csv_data),
                selected = names(values$csv_data)[1],
                multiple = FALSE)
  })

  output$ui_csv_dep_vars <- renderUI({
    selectInput(inputId = 'dependent_variable',
                label = h4("Select a variable to forecast"),
                choices  = names(values$csv_data),
                selected = names(values$csv_data)[2],
                multiple = FALSE)
  })

  # 02.01 ==== LOAD SAMPLE TIME SERIES ===== ----

  # 02.01.01 Update time series selection ----
  update_ts_selection <- reactive({
    if(input$is_csv && is_csv_ok()){
      input$dependent_variable
    } else {
      input$ts_selection
    }
  })

  # 02.01.02 Time series name as header ----
  output$ts_name <- renderText({
      update_ts_selection()
  })

  # 02.01.03 Get time series data ----
  ts_data_tbl <- reactive({

    if(input$is_csv){
      values$csv_data %>%
        csv_as_ts(index = input$time_variable,
                  value = input$dependent_variable)
    } else {
      update_ts_selection() %>% get_ts_data()
    }
  })

  # 02.01.04 Pull time series description ----
  output$ts_description <- reactive({
    if(input$is_csv){
      NULL
    } else {
      ts_data_tbl() %>% pull_ts_desc()
    }

  })


  # 02.01 ==== TIME SERIES ANALYSIS ===== ----

  # 02.01.05 Plot source data ----
  output$dataviz_plot <- renderPlotly({
    ts_data_tbl() %>% plot_ts_data()
  })

  # 02.01.06 Time series summary ----
  output$summary_stats <- renderFormattable({
    ts_data_tbl() %>% ts_summary() %>% formattable(list())
  })

  # 02.01.07 Autocorrelation plot ----
  output$ac_plots <- renderPlotly ({
    ts_data_tbl() %>% plot_acf()
  })

  # 02.01.08 Train proportion ----
  train_prop <- reactive({
    input$train_prop/100
  })

  # 02.01.09 Test proportion ----
  test_prop <- reactive({
    1 - train_prop()
  })

  # 02.01.10 Train/Test split ----
  split_data <- reactive({
    ts_data_tbl() %>% split_ts(train_prop = train_prop())
  })

  # 02.01.11 Plot split data ----
  output$ts_split_plot <- renderPlotly ({
    split_data() %>% plot_ts_split()
  })

  # 02.01.12 Select forecasting method ----
  output$train_method <- reactive({
    if(input$is_autoforecast){
      "AutoForecast model(s) specification"
    } else {
      input$method
    }
  })

  output$forecast_method <- reactive({
    input$method
  })


  # 02.02 ====  MANUAL TRAINING ==== ----

  set_formula <- reactive({

    req(input$method)

    if (input$method == "ARIMA"){

      # pdq(p, d, q) + PDQ(P, D, Q, period=s)
      formula_text <- paste0("pdq(", input$p,",", input$d, ",", input$q,
                             ") + PDQ(", input$P, ",", input$D, ",", input$Q,
                             ", period= ", input$s,")")

    } else if(input$method == "Neural Network Autoregression") {

      # AR(p = 1, P = 1, period = 4)
      formula_text <-  paste0("AR(p=", input$nnar_p, ", P=", input$nnar_P,
                              ", period=", input$nnar_period,")")
    }
  })

  # 02.02.01 Model's formula ----
  output$train_formula <- renderText({
    if(input$is_autoforecast){
      NULL
    } else {
      set_formula()
    }

  })

  output$forecast_formula <- renderText({
    set_formula()
  })

  # 02.01.02 Train model ----
  train_model <- eventReactive(input$train_button, {

    withProgress(message = "Training model", value = 0.5, {

      if(input$method == "ARIMA"){

        fit_arima(split_data(), formula = set_formula())

      } else if(input$method == "Neural Network Autoregression"){

        fit_NNAR(split_data(), formula = set_formula(), n_networks = input$nnar_n)

      } else if(input$method == "Long Short-Term Memory Network"){

        # Clear Keras session
        clear_keras_session()

        # Train model
        fit_lstm(
          # Input data
          train_data = split_data() %>% training(),
          lags = as.integer(input$lstm_lags),
          # Model training
          epochs =  as.integer(input$lstm_epochs),
          batch_size = as.integer(input$lstm_batch_size),
          # Hidden layer
          neurons = as.integer(input$lstm_neurons),
          activation = as.character(input$lstm_activation_function),
          # Optimization algorithm
          optimizer = as.character(input$lstm_optimizer),
          loss = as.character(input$lstm_loss)
        )

      }
    })
  })


  # 02.01.03 Train report ----
  output$model_report <- renderPrint({

    if(input$is_autoforecast){
      auto_train()
    } else if(input$method != "Long Short-Term Memory Network"){
      train_model() %>% report()
    } else if(input$method == "Long Short-Term Memory Network"){
      #train_model()
      train_model() %>% lsmt_summary_report()
    }
  })


  # 02.01.04 Train detail ----
  output$train_detail <- renderTable({

    if(input$is_autoforecast){
      auto_train() %>% autoforecast_model_detail(arima_check = input$arima_check,
                                                 nnar_check = input$nnar_check
                                                 #lstm_check = input$lstm_check
                                                 )
    } else if(input$method != "Neural Network Autoregression"){
      train_model() %>% model_detail(input$method)
    }

  })

  # 02.01.05 Train detail header----
  output$train_detail_header <- reactive({
    if((input$is_autoforecast && input$arima_check && !input$nnar_check)# && !input$lstm_check)
       || (!input$is_autoforecast && input$method == "ARIMA")){
      "Model parameters"
    } else if(input$is_autoforecast || (!input$is_autoforecast && input$method == "Neural Network Autoregression")){
      "Information criterion"
    }
  })

  # 02.03 ====  MANUAL FORECASTING ==== ----

  # 02.03.01 Forecast Horizon  ----
  forecast_horizon <- reactive({
    ceiling(length(ts_data_tbl()) * test_prop())
  })

  # 02.03.02 Manual Forecast  ----
  run_forecast <- eventReactive(input$run_forecast_button, {

      if(input$method == "ARIMA"){

        forecast_arima(arima_model = train_model(),
                       horizon = forecast_horizon())

      } else if(input$method == "Neural Network Autoregression"){

        forecast_NNAR(nnar_model = train_model(),
                      horizon = forecast_horizon(),
                      simulate = input$nnar_pred_intervals)

      } else if(input$method == "Long Short-Term Memory Network"){

        forecast_lstm(lstm_model = train_model(),
                      test_data = split_data() %>% testing(),
                      lags = as.integer(input$lstm_lags),
                      batch_size = as.integer(input$lstm_batch_size))
      }
  })

  # 02.03.03 Plot Forecast ----
  output$forecast_plot <- renderPlot({

    if(input$is_autoforecast){

      auto_forecast() %>% plot_forecast(ts_data = ts_data_tbl())

      } else if (input$method != "Long Short-Term Memory Network"){

      # For ARIMA and NNAR forecast
      run_forecast() %>% plot_forecast(ts_data = ts_data_tbl())

    } else if (input$method == "Long Short-Term Memory Network"){

      run_forecast() %>%
        prepare_lstm_test_tbl(
          timesteps = as.integer(input$lstm_lags),
          test_data = split_data() %>% testing()
        ) %>%
        prepare_lstm_data_plot(
          train_data = split_data() %>% training()
        ) %>%
        plot_time_series(index, value, .title = "", .color_var = name,
                         .line_size = 1, .smooth = FALSE, .interactive = FALSE) +
        theme_tq(base_size = 16) +
        theme(panel.border = element_rect(colour = "lightgrey", fill = NA),
              legend.position = "right") +
        labs(x = "", y = "") +
        scale_y_continuous(breaks = pretty_breaks()) +
        coord_cartesian(expand = FALSE) +
        guides(colour = guide_legend(title = "Forecast"))
    }
  })

  # 02.03.04 Forecast accuracy ----
  output$accuracy_report <- renderTable({
    if(input$is_autoforecast){
      auto_forecast() %>% compute_accuracy(ts_data = ts_data_tbl())
    } else if(input$method != "Long Short-Term Memory Network"){
      run_forecast() %>% compute_accuracy(ts_data = ts_data_tbl())
    } else {
      run_forecast() %>%
        prepare_lstm_test_tbl(
          timesteps = as.integer(input$lstm_lags),
          test_data = split_data() %>% testing()
        ) %>%
      compute_lsmt_accuracy()
    }


  })

  # 02.03.05 Forecast accuracy header ----
  output$accuracy_report_header <- renderText({
    if(input$is_autoforecast){
      header <- NULL
    } else {
      header <- paste(input$method, set_formula())
    }
  })

  # 02.04 ====  AUTOFORECAST ==== ----

  # 02.04.01 AutoTrain ----
  auto_train <- eventReactive(input$auto_forecast_button, {
    withProgress(message = "Training models", value = 0.5, {
      split_data() %>%
        autofit(arima_check = input$arima_check,
                nnar_check  = input$nnar_check
                #lstm_check  = input$lstm_check
                )
    })
  })

  # 02.04.02 AutoForecast ----
  auto_forecast <- reactive({
    auto_train() %>% autoforecast(horizon = forecast_horizon())
  })

  # 02.05 Output options ----
  outputOptions(output, "csv_cols", suspendWhenHidden = FALSE)

# End bracket do not delete
}

# RUN APP ----
shinyApp(ui = ui, server = server)


