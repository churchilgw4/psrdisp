import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt

start = time.time()

def dm_estimate(psrname, timfile, ephemeris):
    
    filename, freq, toa, toaerr, telescope = [], [], [], [], []
    with open(timfile, 'r') as file:
        lin = file.readlines()
    
    for line in lin[2:]:
        columns = line.split()
        filename.append(columns[0])
        freq.append(float(columns[1]))
        toa.append(float(columns[2]))
        toaerr.append(float(columns[3]))
        telescope.append((columns[4]))
        
    filename_arr = np.array(filename)
    freq_arr = np.array(freq)
    toa_arr = np.array(toa)
    toaerr_arr = np.array(toaerr)
    telescope_arr = np.array(telescope)
   
	# -----------------------------------------------------------------------------------------------------------

    condition1 = toaerr_arr != 0.0
    filenamenew = np.extract(condition1, filename_arr)
    freqnew = np.extract(condition1, freq_arr)
    toanew = np.extract(condition1, toa_arr)
    toaerrnew = np.extract(condition1, toaerr_arr)
    telescopenew = np.extract(condition1, telescope_arr)
    
    range_1_indicesN = np.where((freqnew >= 300) & (freqnew <= 505))
    range_2_indicesN = np.where((freqnew >= 550) & (freqnew <= 855))
    range_3_indicesN = np.where((freqnew >= 1160) & (freqnew <= 1465))
    
    fr_b31 = freqnew[range_1_indicesN]
    filename_b31 = filenamenew[range_1_indicesN]
    toa_b31 = toanew[range_1_indicesN]
    toaerr_b31 = toaerrnew[range_1_indicesN]
    telescope_b31 = telescopenew[range_1_indicesN]

    # Taking the filtering condition to be beyond 5-sigma level for Band 3
    conditionb3 = toaerr_b31 < 5*np.median(toaerr_b31)
    
    fr_b32 = np.extract(conditionb3, fr_b31)
    filename_b32 = np.extract(conditionb3, filename_b31)
    toa_b32 = np.extract(conditionb3, toa_b31)
    toaerr_b32 = np.extract(conditionb3, toaerr_b31)
    telescope_b32 = np.extract(conditionb3, telescope_b31)
    
    fr_b41 = freqnew[range_2_indicesN]
    filename_b41 = filenamenew[range_2_indicesN]
    toa_b41 = toanew[range_2_indicesN]
    toaerr_b41 = toaerrnew[range_2_indicesN]
    telescope_b41 = telescopenew[range_2_indicesN]

    # Taking the filtering condition to be beyond 5-sigma level for Band 4
    conditionb4 = toaerr_b41 < 5*np.median(toaerr_b41)
    
    fr_b42 = np.extract(conditionb4,fr_b41)
    filename_b42 = np.extract(conditionb4,filename_b41)
    toa_b42 = np.extract(conditionb4,toa_b41)
    toaerr_b42 = np.extract(conditionb4,toaerr_b41)
    telescope_b42 = np.extract(conditionb4,telescope_b41)
    
    fr_b51 = freqnew[range_3_indicesN]
    filename_b51 = filenamenew[range_3_indicesN]
    toa_b51 = toanew[range_3_indicesN]
    toaerr_b51 = toaerrnew[range_3_indicesN]
    telescope_b51 = telescopenew[range_3_indicesN]

    # Taking the filtering condition to be beyond 5-sigma level for Band 5
    conditionb5 = toaerr_b51 < 5*np.median(toaerr_b51)
    
    fr_b52 = np.extract(conditionb5,fr_b51)
    filename_b52 = np.extract(conditionb5,filename_b51)
    toa_b52 = np.extract(conditionb5,toa_b51)
    toaerr_b52 = np.extract(conditionb5,toaerr_b51)
    telescope_b52 = np.extract(conditionb5,telescope_b51)

    # -----------------------------------------------------------------------------------------------------------

    filenamenew1 = np.concatenate((filename_b32, filename_b42, filename_b52))
    freqnew1 = np.concatenate((fr_b32, fr_b42, fr_b52))
    toanew1 = np.concatenate((toa_b32, toa_b42, toa_b52))
    toaerrnew1 = np.concatenate((toaerr_b32, toaerr_b42, toaerr_b52))
    telescopenew1 = np.concatenate((telescope_b32, telescope_b32, telescope_b52))
    
    tempfile1 = "temp.tim"
    f = open(tempfile1, "w+")
    head="FORMAT 1\n"
    f.write('%s' % head)
    for i in range(0,np.size(freqnew1)):
        f.write('%s %.8f %.18f %.6f %s \n' % (filenamenew1[i], freqnew1[i], 
                toanew1[i], toaerrnew1[i], telescopenew1[i]))
    f.close()

    tmp = os.popen("tempo2 -output general2 -f %s %s -s \"1111111 {freq} {pre} {err}\n\" | grep '1111111'" 
                    % (ephemeris, tempfile1)).read()
    
    os.remove(tempfile1)
    
    # Extracting the data from general2 output
    tmp1 = tmp.split('\n')
    
    freq = []
    residual =[]
    resid_err = []
    for entry in tmp1:
        # Check if the string is not empty
        if entry:                                 
            parts = entry.split()
            freq.append(float(parts[1]))
            residual.append(float(parts[2]))
            resid_err.append(float(parts[3]))
        
    # Convert the list to a numpy array
    freq_array = np.array(freq)
    residual_array = np.array(residual)
    residerr_array = np.array(resid_err)
    residerr_array /= 1e+6
    
    range_1_indices = np.where((freq_array >= 300) & (freq_array <= 505))
    range_2_indices = np.where((freq_array >= 550) & (freq_array <= 855))
    range_3_indices = np.where((freq_array >= 1160) & (freq_array <= 1465))
    
    # Band 3
    x1 = freq_array[range_1_indices]
    y1 = residual_array[range_1_indices]
    e1 = residerr_array[range_1_indices]
    toanew21 = toanew1[range_1_indices]
    toaerrnew21 = toaerrnew1[range_1_indices]
    filenamenew21 = filenamenew1[range_1_indices]
    telescopenew21 = telescopenew1[range_1_indices]
    
    # Band 4
    x2 = freq_array[range_2_indices]
    y2 = residual_array[range_2_indices]
    e2 = residerr_array[range_2_indices]
    toanew22 = toanew1[range_2_indices]
    toaerrnew22 = toaerrnew1[range_2_indices]
    filenamenew22 = filenamenew1[range_2_indices]
    telescopenew22 = telescopenew1[range_2_indices]
    
    # Band 5
    x3 = freq_array[range_3_indices]
    y3 = residual_array[range_3_indices]
    e3 = residerr_array[range_3_indices]
    toanew23 = toanew1[range_3_indices]
    toaerrnew23 = toaerrnew1[range_3_indices]
    filenamenew23 = filenamenew1[range_3_indices]
    telescopenew23 = telescopenew1[range_3_indices]
    
    xn1 = x1
    xn2 = x2
    xn3 = x3
    
    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    x3 = x3.reshape(-1, 1)
    
    from sklearn import linear_model
    from sklearn.linear_model import HuberRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    
    y1 *= 1e+6 
    y2 *= 1e+6
    y3 *= 1e+6
    
    print("Trying to remove outliers using Huber regression...")
        
    if (y1.size != 0 and  y2.size == 0 and y3.size == 0):
        toashift1 = (np.min(y1)*-1.5)
        y1 += toashift1
        e1 = e1*1e+6
        model1 = make_pipeline(PolynomialFeatures(2), HuberRegressor())
        try:
            model1.fit(x1, y1, huberregressor__sample_weight=np.ravel(1./e1))
            y_pred1 = model1.predict(x1)
            residuals1 = y1 - y_pred1
            median1 = np.median(residuals1)
            MAD1 = np.median(np.abs(residuals1-np.median(residuals1)))/0.6744897501960817
            condition1 = (residuals1 > median1 - 5*MAD1) & (residuals1 < median1 + 5*MAD1)
            
            freqf1 = np.extract(condition1, x1)		
            toaf1 = np.extract(condition1, toanew21)
            toaerrf1 = np.extract(condition1, toaerrnew21)	
            filenamef1 = np.extract(condition1, filenamenew21)
            telescopef1 = np.extract(condition1, telescopenew21)
            
            print("Successfully removed outliers using Huber regression in B3.")
    
        except ValueError as ve1:
            print(f"HuberRegressor failed with error: {ve1}")
            print("Moving without Huber Regression in B3.")
            freqf1 = xn1
            toaf1 = toanew21
            toaerrf1 = toaerrnew21
            filenamef1 = filenamenew21
            telescopef1 = telescopenew21
    
        filtered_array1 = [filenamef1, freqf1, toaf1, toaerrf1, telescopef1]
        filtered_array = np.array(filtered_array1)			
                
        dm3, dme3, chi3 = dmfit(filenamef1, freqf1, toaf1, toaerrf1, telescopef1, ephemeris)
        
        dm = [dm3]
        dme = [dme3]
        chi = [chi3]
    
    if (y1.size == 0 and y2.size != 0 and y3.size == 0):
        toashift2 = (np.min(y2)*-1.5)
        y2 += toashift2
        e2 = e2*1e+6
        model2 = make_pipeline(PolynomialFeatures(2), HuberRegressor())
        try:
            model2.fit(x2, y2, huberregressor__sample_weight=np.ravel(1./e2))
            y_pred2 = model2.predict(x2)
            residuals2 = y2 - y_pred2
            median2 = np.median(residuals2)
            MAD2 = np.median(np.abs(residuals2-np.median(residuals2)))/0.6744897501960817
            condition2 = (residuals2 > median2 - 5*MAD2) & (residuals2 < median2 + 5*MAD2)
    
            freqf2 = np.extract(condition2,x2)
            toaf2 = np.extract(condition2,toanew22)
            toaerrf2 = np.extract(condition2,toaerrnew22)	
            filenamef2 = np.extract(condition2,filenamenew22)
            telescopef2 = np.extract(condition2,telescopenew22)
            print("Successfully removed outliers using Huber regression in B4.")
            
        except ValueError as ve2:
            print(f"HuberRegressor failed with error: {ve2}")
            print("Moving without Huber Regression in B4.")
            freqf2 = xn2
            toaf2 = toanew22
            toaerrf2 = toaerrnew22
            filenamef2 = filenamenew22
            telescopef2 = telescopenew22
            
        filtered_array1 = [filenamef2, freqf2, toaf2, toaerrf2, telescopef2]
        filtered_array = np.array(filtered_array1)		
        
        dm4, dme4, chi4 = dmfit(filenamef2, freqf2, toaf2, toaerrf2, telescopef2, ephemeris)
        
        dm = [dm4]
        dme = [dme4]
        chi = [chi4]
    
    if (y1.size == 0 and y2.size == 0 and y3.size != 0):
        toashift3 = (np.min(y3)*-1.5)
        y3 += toashift3
        e3 = e3*1e+6
        model3 = make_pipeline(PolynomialFeatures(2), HuberRegressor())
        try:
            model3.fit(x3, y3, huberregressor__sample_weight=np.ravel(1./e3))
            y_pred3 = model3.predict(x3)
            residuals3 = y3 - y_pred3
            median3 = np.median(residuals3)
            MAD3 = np.median(np.abs(residuals3-np.median(residuals3)))/0.6744897501960817
            condition3 = (residuals3 > median3 - 5*MAD3) & (residuals3 < median3 + 5*MAD3)
    
            freqf3 = np.extract(condition3,x3)
            toaf3 = np.extract(condition3,toanew23)
            toaerrf3 = np.extract(condition3,toaerrnew23)	
            filenamef3 = np.extract(condition3,filenamenew23)
            telescopef3 = np.extract(condition3,telescopenew23)
            print("Successfully removed outliers using Huber regression in B5.")
            
        except ValueError as ve3:
            print(f"HuberRegressor failed with error: {ve3}")
            print("Moving without Huber Regression in B5.")
            freqf3 = xn3
            toaf3 = toanew23
            toaerrf3 = toaerrnew23
            filenamef3 = filenamenew23
            telescopef3 = telescopenew23
            
        filtered_array1 = [filenamef3, freqf3, toaf3, toaerrf3, telescopef3]
        filtered_array = np.array(filtered_array1)	
        
        dm5, dme5, chi5 = dmfit(filenamef3, freqf3, toaf3, toaerrf3, telescopef3, ephemeris)
    
        dm = [dm5]
        dme = [dme5]
        chi = [chi5]
    
    if (y2.size == 0 and y1.size !=0 and y3.size !=0):
        
        toashift1 = (np.min(y1)*-1.5)
        toashift3 = (np.min(y3)*-1.5)
        y1 += toashift1
        y3 += toashift3
        e1 = e1*1e+6
        e3 = e3*1e+6
    
        model1 = make_pipeline(PolynomialFeatures(2), HuberRegressor())
        model3 = make_pipeline(PolynomialFeatures(2), HuberRegressor())
    
        try:
            model1.fit(x1, y1, huberregressor__sample_weight=np.ravel(1./e1))
            y_pred1 = model1.predict(x1)
            residuals1 = y1 - y_pred1
            median1 = np.median(residuals1)
            MAD1 = np.median(np.abs(residuals1-np.median(residuals1)))/0.6744897501960817
            condition1 = (residuals1 > median1 - 5*MAD1) & (residuals1 < median1 + 5*MAD1)
            
            freqf1 = np.extract(condition1, x1)		
            toaf1 = np.extract(condition1, toanew21)
            toaerrf1 = np.extract(condition1, toaerrnew21)	
            filenamef1 = np.extract(condition1, filenamenew21)
            telescopef1 = np.extract(condition1, telescopenew21)
            print("Successfully removed outliers using Huber regression in B3.")
            
        except ValueError as ve1:
            print(f"HuberRegressor failed with error: {ve1}")
            print("Moving without Huber Regression in B3.")
            freqf1 = xn1
            toaf1 = toanew21
            toaerrf1 = toaerrnew21
            filenamef1 = filenamenew21
            telescopef1 = telescopenew21
    
        try:
            model3.fit(x3, y3, huberregressor__sample_weight=np.ravel(1./e3))
            y_pred3 = model3.predict(x3)
            residuals3 = y3 - y_pred3
            median3 = np.median(residuals3)
            MAD3 = np.median(np.abs(residuals3-np.median(residuals3)))/0.6744897501960817
            condition3 = (residuals3 > median3 - 5*MAD3) & (residuals3 < median3 + 5*MAD3)
    
            freqf3 = np.extract(condition3, x3)
            toaf3 = np.extract(condition3, toanew23)
            toaerrf3 = np.extract(condition3, toaerrnew23)	
            filenamef3 = np.extract(condition3, filenamenew23)
            telescopef3 = np.extract(condition3, telescopenew23)
            print("Successfully removed outliers using Huber regression in B5.")
            
        except ValueError as ve3:
            print(f"HuberRegressor failed with error: {ve3}")
            print("Moving without Huber Regression in B5.")
            freqf3 = xn3
            toaf3 = toanew23
            toaerrf3 = toaerrnew23
            filenamef3 = filenamenew23
            telescopef3 = telescopenew23
        
        freq3p5 = np.concatenate((freqf1, freqf3))
        toaf3p5 = np.concatenate((toaf1, toaf3))
        toaerrf3p5 = np.concatenate((toaerrf1, toaerrf3))
        filenamef3p5 = np.concatenate((filenamef1, filenamef3))
        telescopef3p5 = np.concatenate((telescopef1, telescopef3))
    
        filtered_array1 = [filenamef3p5, freq3p5, toaf3p5, toaerrf3p5, telescopef3p5]
        filtered_array = np.array(filtered_array1)
    
        dm3, dme3, chi3 = dmfit(filenamef1, freqf1, toaf1, toaerrf1, telescopef1, ephemeris)
        dm5, dme5, chi5 = dmfit(filenamef3, freqf3, toaf3, toaerrf3, telescopef3, ephemeris)
        dm3p5, dme3p5, chi3p5 = dmfit(filenamef3p5, freq3p5, toaf3p5, toaerrf3p5, telescopef3p5, ephemeris)
        
        dm = [dm3p5, dm3, dm5]
        dme = [dme3p5, dme3, dme5]
        chi = [chi3p5, chi3, chi5]
    
    if (y3.size == 0 and y1.size !=0 and y2.size != 0):
        
        toashift1 = (np.min(y1)*-1.5)
        toashift2 = (np.min(y2)*-1.5)
        y1 += toashift1
        y2 += toashift2
        e1 = e1*1e+6
        e2 = e2*1e+6
    
        model1 = make_pipeline(PolynomialFeatures(2), HuberRegressor())
        model2 = make_pipeline(PolynomialFeatures(2), HuberRegressor())
    
        try:
            model1.fit(x1,y1,huberregressor__sample_weight=np.ravel(1./e1))
            y_pred1 = model1.predict(x1)
            residuals1 = y1 - y_pred1
            median1 = np.median(residuals1)
            MAD1 = np.median(np.abs(residuals1-np.median(residuals1)))/0.6744897501960817
            condition1 = (residuals1 > median1 - 5*MAD1) & (residuals1 < median1 + 5*MAD1)
            
            freqf1 = np.extract(condition1,x1)		
            toaf1 = np.extract(condition1,toanew21)
            toaerrf1 = np.extract(condition1,toaerrnew21)	
            filenamef1 = np.extract(condition1,filenamenew21)
            telescopef1 = np.extract(condition1,telescopenew21)
            print("Successfully removed outliers using Huber regression in B3.")
            
        except ValueError as ve1:
            print(f"HuberRegressor failed with error: {ve1}")
            print("Moving without Huber Regression in B3.")
            freqf1 = xn1
            toaf1 = toanew21
            toaerrf1 = toaerrnew21
            filenamef1 = filenamenew21
            telescopef1 = telescopenew21
    
        try:
            model2.fit(x2,y2,huberregressor__sample_weight=np.ravel(1./e2))
            y_pred2 = model2.predict(x2)
            residuals2 = y2 - y_pred2
            median2 = np.median(residuals2)
            MAD2 = np.median(np.abs(residuals2-np.median(residuals2)))/0.6744897501960817
            condition2 = (residuals2 > median2 - 5*MAD2) & (residuals2 < median2 + 5*MAD2)
    
            freqf2 = np.extract(condition2,x2)
            toaf2 = np.extract(condition2,toanew22)
            toaerrf2 = np.extract(condition2,toaerrnew22)	
            filenamef2 = np.extract(condition2,filenamenew22)
            telescopef2 = np.extract(condition2,telescopenew22)
            print("Successfully removed outliers using Huber regression in B4.")
            
        except ValueError as ve2:
            print(f"HuberRegressor failed with error: {ve2}")
            print("Moving without Huber Regression in B4.")
            freqf2 = xn2
            toaf2 = toanew22
            toaerrf2 = toaerrnew22
            filenamef2 = filenamenew22
            telescopef2 = telescopenew22
        
        freq3p4 = np.concatenate((freqf1, freqf2))
        toaf3p4 = np.concatenate((toaf1, toaf2))
        toaerrf3p4 = np.concatenate((toaerrf1, toaerrf2))
        filenamef3p4 = np.concatenate((filenamef1, filenamef2))
        telescopef3p4 = np.concatenate((telescopef1, telescopef2))
    
        filtered_array1 = [filenamef3p4, freq3p4, toaf3p4, toaerrf3p4, telescopef3p4]
        filtered_array = np.array(filtered_array1)
    
        dm3, dme3, chi3 = dmfit(filenamef1, freqf1, toaf1, toaerrf1, telescopef1, ephemeris)
        dm4, dme4, chi4 = dmfit(filenamef2, freqf2, toaf2, toaerrf2, telescopef2, ephemeris)
        dm3p4, dme3p4, chi3p4 = dmfit(filenamef3p4, freq3p4, toaf3p4, toaerrf3p4, telescopef3p4, ephemeris)
    
        dm = [dm3p4, dm3, dm4]
        dme = [dme3p4, dme3, dme4]
        chi = [chi3p4, chi3, chi4]
    
    if (y1.size == 0 and y2.size !=0 and y3.size != 0):
        
        toashift2 = (np.min(y2)*-1.5)
        toashift3 = (np.min(y3)*-1.5)
        y2 += toashift2
        y3 += toashift3
        e3 = e3*1e+6
        e2 = e2*1e+6
    
        model3 = make_pipeline(PolynomialFeatures(2), HuberRegressor())
        model2 = make_pipeline(PolynomialFeatures(2), HuberRegressor())
    
        try:
            model3.fit(x3,y3,huberregressor__sample_weight=np.ravel(1./e3))
            y_pred3 = model3.predict(x3)
            residuals3 = y3 - y_pred3
            median3 = np.median(residuals3)
            MAD3 = np.median(np.abs(residuals3-np.median(residuals3)))/0.6744897501960817
            condition3 = (residuals3 > median3 - 5*MAD3) & (residuals3 < median3 + 5*MAD3)
            
            freqf3 = np.extract(condition3,x3)		
            toaf3 = np.extract(condition3,toanew23)
            toaerrf3 = np.extract(condition3,toaerrnew23)	
            filenamef3 = np.extract(condition3,filenamenew23)
            telescopef3 = np.extract(condition3,telescopenew23)
            print("Successfully removed outliers using Huber regression in B5.")
            
        except ValueError as ve3:
            print(f"HuberRegressor failed with error: {ve3}")
            print("Moving without Huber Regression in B5.")
            freqf3 = xn3
            toaf3 = toanew23
            toaerrf3 = toaerrnew23
            filenamef3 = filenamenew23
            telescopef3 = telescopenew23
    
        try:
            model2.fit(x2,y2,huberregressor__sample_weight=np.ravel(1./e2))
            y_pred2 = model2.predict(x2)
            residuals2 = y2 - y_pred2
            median2 = np.median(residuals2)
            MAD2 = np.median(np.abs(residuals2-np.median(residuals2)))/0.6744897501960817
            condition2 = (residuals2 > median2 - 5*MAD2) & (residuals2 < median2 + 5*MAD2)
    
            freqf2 = np.extract(condition2,x2)
            toaf2 = np.extract(condition2,toanew22)
            toaerrf2 = np.extract(condition2,toaerrnew22)	
            filenamef2 = np.extract(condition2,filenamenew22)
            telescopef2 = np.extract(condition2,telescopenew22)
            print("Successfully removed outliers using Huber regression in B4.")
            
        except ValueError as ve2:
            print(f"HuberRegressor failed with error: {ve2}")
            print("Moving without Huber Regression in B4.")
            freqf2 = xn2
            toaf2 = toanew22
            toaerrf2 = toaerrnew22
            filenamef2 = filenamenew22
            telescopef2 = telescopenew22
        
        freq4p5 = np.concatenate((freqf2, freqf3))
        toaf4p5 = np.concatenate((toaf2, toaf3))
        toaerrf4p5 = np.concatenate((toaerrf2, toaerrf3))
        filenamef4p5 = np.concatenate((filenamef2, filenamef3))
        telescopef4p5 = np.concatenate((telescopef2, telescopef3))
    
        filtered_array1 = [filenamef4p5, freq4p5, toaf4p5, toaerrf4p5, telescopef4p5]
        filtered_array = np.array(filtered_array1)
    
        dm5, dme5, chi5 = dmfit(filenamef3, freqf3, toaf3, toaerrf3, telescopef3, ephemeris)
        dm4, dme4, chi4 = dmfit(filenamef2, freqf2, toaf2, toaerrf2, telescopef2, ephemeris)
        dm4p5, dme4p5, chi4p5 = dmfit(filenamef4p5, freq4p5, toaf4p5, toaerrf4p5, telescopef4p5, ephemeris)
    
        dm = [dm4p5, dm4, dm5]
        dme = [dme4p5, dme4, dme5]
        chi = [chi4p5, chi4, chi5]
    
    if (y1.size != 0 and y2.size != 0 and y3.size != 0):
        
        toashift1 = (np.min(y1)*-1.5)
        toashift2 = (np.min(y2)*-1.5)
        toashift3 = (np.min(y3)*-1.5)
        y1 += toashift1
        y2 += toashift2
        y3 += toashift3
        e1 = e1*1e+6
        e2 = e2*1e+6
        e3 = e3*1e+6
    
        model1 = make_pipeline(PolynomialFeatures(2), HuberRegressor())
        model2 = make_pipeline(PolynomialFeatures(2), HuberRegressor())
        model3 = make_pipeline(PolynomialFeatures(2), HuberRegressor())
    
        try:
            model1.fit(x1,y1,huberregressor__sample_weight=np.ravel(1./e1))
            y_pred1 = model1.predict(x1)
            residuals1 = y1 - y_pred1
            median1 = np.median(residuals1)
            MAD1 = np.median(np.abs(residuals1-np.median(residuals1)))/0.6744897501960817
            condition1 = (residuals1 > median1 - 5*MAD1) & (residuals1 < median1 + 5*MAD1)
            
            freqf1 = np.extract(condition1,x1)		
            toaf1 = np.extract(condition1,toanew21)
            toaerrf1 = np.extract(condition1,toaerrnew21)	
            filenamef1 = np.extract(condition1,filenamenew21)
            telescopef1 = np.extract(condition1,telescopenew21)
            print("Successfully removed outliers using Huber regression in B3.")
            
        except ValueError as ve1:
            print(f"HuberRegressor failed with error: {ve1}")
            print("Moving without Huber Regression in B3.")
            freqf1 = xn1
            toaf1 = toanew21
            toaerrf1 = toaerrnew21
            filenamef1 = filenamenew21
            telescopef1 = telescopenew21
            
        try:
            model2.fit(x2,y2,huberregressor__sample_weight=np.ravel(1./e2))
            y_pred2 = model2.predict(x2)
            residuals2 = y2 - y_pred2
            median2 = np.median(residuals2)
            MAD2 = np.median(np.abs(residuals2-np.median(residuals2)))/0.6744897501960817
            condition2 = (residuals2 > median2 - 5*MAD2) & (residuals2 < median2 + 5*MAD2)
    
            freqf2 = np.extract(condition2,x2)
            toaf2 = np.extract(condition2,toanew22)
            toaerrf2 = np.extract(condition2,toaerrnew22)	
            filenamef2 = np.extract(condition2,filenamenew22)
            telescopef2 = np.extract(condition2,telescopenew22)
            print("Successfully removed outliers using Huber regression in B4.")
            
        except ValueError as ve2:
            print(f"HuberRegressor failed with error: {ve2}")
            print("Moving without Huber Regression in B4.")
            freqf2 = xn2
            toaf2 = toanew22
            toaerrf2 = toaerrnew22
            filenamef2 = filenamenew22
            telescopef2 = telescopenew22
    
        try:
            model3.fit(x3,y3,huberregressor__sample_weight=np.ravel(1./e3))
            y_pred3 = model3.predict(x3)
            residuals3 = y3 - y_pred3
            median3 = np.median(residuals3)
            MAD3 = np.median(np.abs(residuals3-np.median(residuals3)))/0.6744897501960817
            condition3 = (residuals3 > median3 - 5*MAD3) & (residuals3 < median3 + 5*MAD3)
    
            freqf3 = np.extract(condition3,x3)
            toaf3 = np.extract(condition3,toanew23)
            toaerrf3 = np.extract(condition3,toaerrnew23)	
            filenamef3 = np.extract(condition3,filenamenew23)
            telescopef3 = np.extract(condition3,telescopenew23)
            print("Successfully removed outliers using Huber regression in B5.")
            
        except ValueError as ve3:
            print(f"HuberRegressor failed with error: {ve3}")
            print("Moving without Huber Regression in B5.")
            freqf3 = xn3
            toaf3 = toanew23
            toaerrf3 = toaerrnew23
            filenamef3 = filenamenew23
            telescopef3 = telescopenew23
    
        freqf3p4 = np.concatenate((freqf1, freqf2))
        toaf3p4 = np.concatenate((toaf1, toaf2))
        toaerrf3p4 = np.concatenate((toaerrf1, toaerrf2))
        filenamef3p4 = np.concatenate((filenamef1, filenamef2))
        telescopef3p4 = np.concatenate((telescopef1, telescopef2))
    
        freqf3p5 = np.concatenate((freqf1, freqf3))
        toaf3p5 = np.concatenate((toaf1, toaf3))
        toaerrf3p5 = np.concatenate((toaerrf1, toaerrf3))
        filenamef3p5 = np.concatenate((filenamef1, filenamef3))
        telescopef3p5 = np.concatenate((telescopef1, telescopef3))
    
        freqf4p5 = np.concatenate((freqf2, freqf3))
        toaf4p5 = np.concatenate((toaf2, toaf3))
        toaerrf4p5 = np.concatenate((toaerrf2, toaerrf3))
        filenamef4p5 = np.concatenate((filenamef2, filenamef3))
        telescopef4p5 = np.concatenate((telescopef2, telescopef3))
    
        freqf3p4p5 = np.concatenate((freqf1, freqf2, freqf3))
        toaf3p4p5 = np.concatenate((toaf1, toaf2, toaf3))
        toaerrf3p4p5 = np.concatenate((toaerrf1, toaerrf2, toaerrf3))
        filenamef3p4p5 = np.concatenate((filenamef1, filenamef2, filenamef3))
        telescopef3p4p5 = np.concatenate((telescopef1, telescopef2, telescopef3))
    
        filtered_array1 = [filenamef3p4p5, freqf3p4p5, toaf3p4p5, toaerrf3p4p5, telescopef3p4p5]
        filtered_array = np.array(filtered_array1)
    
        dm3, dme3, chi3 = dmfit(filenamef1, freqf1, toaf1, toaerrf1, telescopef1, ephemeris)
        dm4, dme4, chi4 = dmfit(filenamef2, freqf2, toaf2, toaerrf2, telescopef2, ephemeris)
        dm5, dme5, chi5 = dmfit(filenamef3, freqf3, toaf3, toaerrf3, telescopef3, ephemeris)
        dm3p4, dme3p4, chi3p4 = dmfit(filenamef3p4, freqf3p4, toaf3p4, toaerrf3p4, telescopef3p4, ephemeris)
        dm3p5, dme3p5, chi3p5 = dmfit(filenamef3p5, freqf3p5, toaf3p5, toaerrf3p5, telescopef3p5, ephemeris)
        dm4p5, dme4p5, chi4p5 = dmfit(filenamef4p5, freqf4p5, toaf4p5, toaerrf4p5, telescopef4p5, ephemeris)
        dm3p4p5, dme3p4p5, chi3p4p5 = dmfit(filenamef3p4p5, freqf3p4p5, toaf3p4p5, toaerrf3p4p5, telescopef3p4p5, ephemeris)
    
        dm = [dm3p4p5, dm3p4, dm3p5, dm4p5, dm3, dm4, dm5]
        dme = [dme3p4p5, dme3p4, dme3p5, dme4p5, dme3, dme4, dme5]
        chi = [chi3p4p5, chi3p4, chi3p5, chi4p5, chi3, chi4, chi5]
    
    return(dm, dme, chi, freq_array, residual_array, residerr_array, filtered_array)

def dmfit(filename, freqs, toas, toaE, tel, ephemeris):
    
    tempfile = filename[0]+"_"+str(freqs[0])+"_toas.txt"
    f = open(tempfile, "w+")
    head="FORMAT 1\n"
    f.write('%s' % head)
    for i in range(freqs.size):
        f.write('%s %.8f %.18f %.6f %s\n' % (filename[i], freqs[i], toas[i], toaE[i], tel[i]))
    f.close()
    awkstr = "-nofit -fit dm | grep 'DM (cm^-3 pc)'| awk \'{print $5,$6}\'"
    dmstr = os.popen("tempo2 -f %s %s %s" % (ephemeris, tempfile, awkstr)).read()
    (dm, dmerr) = dmstr.split()
    dmval = float(dm)
    dmverr = float(dmerr)
    
    # Performing the fit again to read the chisquare
    chisqstr = os.popen("tempo2 -f %s %s -nofit -fit dm | grep 'Fit Chisq'| awk \'{print $9}\'" % (ephemeris, tempfile)).read()
    fitchisq = float(chisqstr)
    os.remove(tempfile)
    
    return(dmval, dmverr, fitchisq)

def plot_dms(psrname, dmfile, tag='Band 3'):
    '''
        Function to plot the DM time series
    '''
    lines = [f.split('\n')[0] for f in open(dmfile, 'r').readlines()]
    toas = [float(line.split()[0]) for line in lines]
    dms = [float(line.split()[1]) for line in lines]
    dmerrs = [float(line.split()[2]) for line in lines]
    
    fig = plt.figure(figsize=(10, 5), dpi=100)
    plt.errorbar(toas, dms, yerr=dmerrs, fmt='o', ms=7, c='crimson', alpha=0.5, 
                 elinewidth=1.0, capsize=2, capthick=0.5, ecolor='k')
    plt.ticklabel_format(useOffset=False)
    plt.grid(ls='dashed')
    _ = plt.xlabel(f'MJD', fontweight='bold', fontsize=13)
    _ = plt.ylabel(rf'DM [pc/cm$^3$]', fontweight='bold', fontsize=13)
    plt.title(rf'{psrname}, {tag} median '+r'$\sigma_{DM}$'+
              rf' = {np.median(dmerrs):.2e} pc/cm$^3$', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'./{psrname}.{tag}.DM.time.series.pdf')

if __name__ == '__main__':
    
    psrname = sys.argv[1]
    timfile = sys.argv[2]
    parfile = sys.argv[3]
    nchan_b3 = int(sys.argv[4])
    nchan_b5 = int(sys.argv[5])

    file = [f.split('\n')[0] for f in open(timfile, 'r').readlines()]
    lines = [f for f in file if not f.startswith('FORMAT') and not f.startswith('MODE')]
    toas = int(len(lines)/(nchan_b3+nchan_b5))

    tempdir = f'./epochwiseToAs'
    try: os.mkdir(tempdir)
    except OSError: pass

    for i in range(toas):
        epochwiseToAsfile = f'{tempdir}/{psrname}.epoch.{i+1}.txt'
        f = open(epochwiseToAsfile, 'w+')
        if i:
            for j in range(nchan_b3+nchan_b5):
                index = i*(nchan_b3+nchan_b5)+j
                f.write(f'{lines[index]}\n')
        else:
            for j in range(nchan_b3+nchan_b5):
                index = j
                f.write(f'{lines[index]}\n')
        f.close()

    dmfile_b35_filename = psrname+".DM.timeseries.B35.txt"
    dmfile_b3_filename = psrname+".DM.timeseries.B3.txt"
    
    if not os.path.isfile(dmfile_b3_filename) and not os.path.isfile(dmfile_b35_filename):
        
        dmfile_b35 = open(dmfile_b35_filename, 'a')
        dmfile_b35.close()
    
        dmfile_b3 = open(dmfile_b3_filename, 'a')
        dmfile_b3.close()
    
        filtered_timfile = psrname+".filtered.toas.txt"
        f = open(filtered_timfile, 'a')
        head="FORMAT 1\nMODE 1\n"
        f.write('%s' % head)
        f.close()
        
        for i in range(toas):
            epochwisetimfile = f'{tempdir}/{psrname}.epoch.{i+1}.txt'
            dm, dmerror, chisq, freq_array, residual_array, residerr_array, filtered_array = dm_estimate(psrname, 
                                                                                                         epochwisetimfile, 
                                                                                                         parfile)
        
            print("Adding the filtered ToAs...")
            filename_fil = filtered_array[0]
            freqs_fil = filtered_array[1].astype(float)
            toas_fil = filtered_array[2].astype(float)
            toaE_fil = filtered_array[3].astype(float)
            tel_fil = filtered_array[4]
    
            f = open(filtered_timfile, 'a')
            for i in range(freqs_fil.size):
                f.write('%s %.7f %.7f %.7f %s\n' % (filename_fil[i], toas_fil[i], toaE_fil[i], freqs_fil[i], tel_fil[i]))
            f.close()
    
            if nchan_b3 and nchan_b5:
                print("Generating the Band 3+5 DM time series...")
                dmfile_b35 = open(psrname+".DM.timeseries.B35.txt", 'a')
                dmfile_b35.write('%.7f %.7f %.7f %.2f\n' %(toas_fil[0], dm[0], dmerror[0], chisq[0]))
                dmfile_b35.close()
                print("Generating the Band 3 DM time series...")
                dmfile_b3 = open(psrname+".DM.timeseries.B3.txt", 'a')
                dmfile_b3.write('%.7f %.7f %.7f %.2f\n' %(toas_fil[1], dm[1], dmerror[1], chisq[1]))
                dmfile_b3.close()
            elif nchan_b3:
                print("Generating the Band 3 DM time series...")
                dmfile_b3 = open(psrname+".DM.timeseries.B3.txt", 'a')
                dmfile_b3.write('%.7f %.7f %.7f %.2f\n' %(toas_fil[0], dm[0], dmerror[0], chisq[0]))
                dmfile_b3.close()
            elif nchan_b5:
                print("Generating the Band 3+5 DM time series...")
                dmfile_b35 = open(psrname+".DM.timeseries.B35.txt", 'a')
                dmfile_b35.write('%.7f %.7f %.7f %.2f\n' %(toas_fil[0], dm[0], dmerror[0], chisq[0]))
                dmfile_b35.close()
            else:
                raise Exception('Both nchan_b3 and nchan_b5 cannot be zero.')

        if nchan_b3:
            plot_dms(psrname, dmfile_b3_filename, 'Band3')
            if nchan_b5:
                plot_dms(psrname, dmfile_b35_filename, 'Band3+5')
    
    else:
        print(f'Directly plotting the DM time series from existing files')
        if nchan_b3:
            plot_dms(psrname, dmfile_b3_filename, 'Band3')
            if nchan_b5:
                plot_dms(psrname, dmfile_b35_filename, 'Band3+5')
        
    end = time.time()
    
    print('\n--------------------------------------------------------------')
    print(f'The process took {(end-start):.4f}s to finish')
    print('--------------------------------------------------------------')

