check_file_extension <- function(file){
  ext <- tools::file_ext(file$datapath)
  return(ext == "csv")
}

parse_index <- function(index){
  try(as.Date(index, tryFormats = c("%Y-%m-%d", "%Y/%m/%d")), silent = TRUE)
  try(as.yearqtr(index), silent = TRUE)
}

coerce_index <- function(index){

  # yearmon and yearqtr are not supported by fable
  # coerce to yearmonth / yearquarter

  # If index's class is yearmon
  if(class(index) == "yearmon"){
    # Coerce yearmonth
    index <- tsibble::yearmonth(index)
  } else if(class(index) == "yearqtr"){
    # Coerce yearquarter
    index <- tsibble::yearquarter(index)
  } else {
    index
  }
}

csv_as_ts <- function(csv_df, index, value){

  csv_df %>%
    select(index = index, value = value) %>%
    mutate(index = parse_index(index)) %>%
    read.zoo() %>%
    as.ts()
}



# Get list of sample time series
get_ts_list <- function() {

  # Import text file with names of selected time series
  selected_ts <- read.delim("selected_time_series.txt")

  # Import meta data from the Time Series Data Library.
  ts_data_library <- meta_tsdl$description %>%
    unlist() %>%
    as.data.frame() %>%
    filter(. %in% selected_ts$ts_name)

  return(as.list(ts_data_library[1]))

}

# Get ts data from user input
get_ts_data <- function(ts_name){
  subset(tsdl, description = ts_name)[[1]]
}


# Clean time series description
pull_ts_desc <- function(ts_data){
  ts_description <- xtsAttributes(ts_data)$description
}

# Time series summary
ts_summary <- function(ts_data){

  summary <- ts_data %>%
    skim() %>%
    select(!(c(skim_variable, skim_type, ts.deltat, ts.line_graph))) %>%
    select(start = ts.start,
           end = ts.end,
           missing_values = n_missing,
           complete_rate,
           frequency = ts.frequency,
           min_value = ts.min,
           max_value = ts.max,
           mean = ts.mean,
           median = ts.median
    )

  return(summary)
}


# Visualize time series
plot_ts_data <- function(ts_data){

  ts_plot <- ts_data %>%
    tk_tbl(rename_index = "date") %>%
    plot_time_series(date, value,
                     .smooth = FALSE,
                     .title = "",
                     .plotly_slider = TRUE)
  return(ts_plot)
}

# Autocorrelation plots
plot_acf <- function(ts_data){

  ac_plot <- tk_tbl(ts_data) %>%
    plot_acf_diagnostics(.date_var = index,
                         .value = value,
                         .lags = 1:31,
                         .show_white_noise_bars = TRUE,
                         .white_noise_line_color = "red",
                         .title = "",
                         .interactive = FALSE) +
    theme_tq()

  ac_plot <- ggplotly(ac_plot, height = 600) %>%
    layout(autosize = TRUE)

  return(ac_plot)
}

# Train-test split using the rsample library
split_ts <- function(ts_data, train_prop = 0.80){

  split_data <- tk_tbl(ts_data) %>%
    initial_time_split(prop = train_prop)

  return(split_data)
}

# Train-test plot using the timetk library
plot_ts_split <- function(split_data){

  train_test_plot <- split_data %>%
    tk_time_series_cv_plan() %>%
    plot_time_series_cv_plan(index, value, .title = "", .interactive = TRUE)

  return(train_test_plot)
}


# Fit ARIMA model using manual specification - Fable library
fit_arima <- function(split_data, formula, is_log = TRUE){

  #Give the user the option to modify is_log flag

  split_data$data$index <- coerce_index(split_data$data$index)

  train <- split_data %>%
    # extract train data from split
    training() %>%
    # coerce to a tsibble
    as_tsibble(index=index) %>%
    model(
      # fit arima model with log transformation
      ARIMA = ARIMA(as.formula(paste0("log(value) ~ 0 +", formula)))
    )
  return(train)
}

# Fit Neural Network Autoregressive
fit_NNAR <- function(split_data, formula, n_networks, scale_inputs = TRUE){

  split_data$data$index <- coerce_index(split_data$data$index)

  train <- split_data %>%
    # extract train data from split
    training() %>%
    # coerce to a tsibble
    as_tsibble(index = index) %>%
    # fit neural network autoregressive
    model(
      NAAR = NNETAR(as.formula(paste0("value ~ ", formula)),
                    n_networks = n_networks,
                    scale_inputs = scale_inputs
      )
    )
  return(train)
}

# Model detail
model_detail <- function(fit_model, method){

  if(method == "ARIMA"){
    detail <- fit_model %>% glance() %>% select(-c(ar_roots, ma_roots))
  } else if(method == "Neural Network Autoregression"){
    detail <- fit_model %>% glance()
  } else if (method == "Long Short-Term Memory Network"){
    # ARIMA and NNAR
    detail <- "LSTM is non-parametric method"
  } else {
    detail <- "Unexpected method delected"
  }
  return(detail)
}

# Forecast arima model
forecast_arima <- function(arima_model, horizon){
  arima_forecast <- arima_model %>% forecast(h = horizon)
  return(arima_forecast)
}

# Forecast NNAR model
forecast_NNAR <- function(nnar_model, horizon, simulate = FALSE){

  nnar_forecast <- nnar_model %>%
    forecast(h = horizon, simulate = simulate)
  return(nnar_forecast)
}

# AutoFit
autofit <- function(split_data, arima_check, nnar_check){ #, lstm_check){

  split_data$data$index <- coerce_index(split_data$data$index)

  train_data <- split_data %>% training() %>% as_tsibble(index = index)

  if(arima_check && !nnar_check){ #&& !lstm_check){
    # ARIMA only
    train_model <- train_data %>%
      model(
        ARIMA = ARIMA(log(value))
      )
  } else if(!arima_check && nnar_check){ #&& !lstm_check){
    # NNAR only
    train_model <- train_data %>%
      model(
        NNAR = NNETAR(box_cox(value, 0.15))
      )
  } else if(arima_check && nnar_check){ # && !lstm_check){
    # ARIMA and NNAR
    train_model <- train_data %>%
      model(
        ARIMA = ARIMA(log(value)),
        NNAR  = NNETAR(box_cox(value, 0.15))
      )
  }

  return(train_model)
}


# AutoForecast - model detail
autoforecast_model_detail <- function(fit_model, arima_check, nnar_check){ #, lstm_check){

  if(arima_check && !nnar_check){ # && !lstm_check){
    # ARIMA only
    detail <- fit_model %>% coef() %>% select(-.model)
  } else if(!arima_check && nnar_check){ # && !lstm_check){
    # NNAR only
    detail <- fit_model %>% glance()
  } else if (arima_check && nnar_check){ # && !lstm_check){
    # ARIMA and NNAR
    detail <- fit_model %>% glance() %>% select(-c(ar_roots, ma_roots))
  }
  return(detail)
}

# AutoForecast
autoforecast <- function(train_model, horizon){
  forecast(train_model, h = horizon)
}

# Plot Forecast
plot_forecast <- function(forecast_data, ts_data){

  is_unimodel <- length(unique(forecast_data$`.model`)) == 1

  if(is_unimodel){
    level <- c(80, 95)
  } else {
    level <- NULL
  }

  plot <- forecast_data %>%
    autoplot(ts_data, level = level, color = "dodgerblue4", size = 1) +
    theme_tq(base_size = 16) +
    theme(panel.border = element_rect(colour = "lightgrey", fill = NA),
          legend.position = "right") +
    labs(x = "", y = "") +
    coord_cartesian(expand = FALSE) +
    guides(colour = guide_legend(title = "Forecast"))

  return(plot)
}

# Forecast accuracy
compute_accuracy <- function(forecast_data, ts_data){
  fabletools::accuracy(forecast_data, as_tsibble(ts_data),
           measures = point_accuracy_measures) %>%
    select(-c(.type, ME, ACF1))
}

prepare_lstm_test_tbl <- function(lstm_prediction, timesteps, test_data){

  test_data %>%
    add_column(forecast = c(rep(NA, times = as.integer(timesteps)), lstm_prediction)) %>%
    drop_na()
}

prepare_lstm_data_plot <- function(lstm_test_tbl, train_data){

  lstm_test_tbl %>%
    bind_rows(
      train_data %>% add_column(forecast = NA)
    ) %>%
    arrange(index) %>%
    select(index, actual = value, forecast) %>%
    pivot_longer(cols = c(actual, forecast)) %>%
    arrange(index, name)
}

compute_lsmt_accuracy <- function(lstm_test_tbl){
  lstm_test_tbl %>%
    summarize(
      RMSE  = rmse_vec(value, forecast),
      MAE   = mae_vec(value, forecast),
      MPE   = mpe_vec(value, forecast),
      MAPE  = mape_vec(value, forecast),
      MASE  = mase_vec(value, forecast)
    )
}














