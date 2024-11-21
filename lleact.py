import decimal
import numpy as np
import pandas as pd
import liquidmodel as lm
from calc_deltagn_wrapper import calc_deltagn_cpp
from scipy.optimize import fsolve, least_squares
from scipy.integrate import simpson


class Context():
    
    def create_mixture(self, liquidmodel):
        self.liquidmodel = liquidmodel


class Calc_binary_critical:
    
    tol_critical = {'NRTL':1.e-13,
                    'Margules':1.e-13,
                    'Wilson':1.e-13,
                    'COSMOSAC2002e':1.e-8,
                    'COSMOSACb':1.e-8,
                    'COSMOSAC_MIT':1.e-5}
    
    def __init__(self, liquidmodelname, mixturename=None, objfunmethod=None, hexmethod=None):
        self.liquidmodelname = liquidmodelname
        self.cbc_lmobject = lm.Context(lm.create(liquidmodelname))
        if mixturename is not None:
            self._mixturename = mixturename
            self.cbc_lmobject.mixture = mixturename
        self.lletype = None
        if objfunmethod is not None:
            self.objfunmethod = objfunmethod
            self.hexmethod = hexmethod
        else:
            if liquidmodelname[:5] == 'COSMO':
                self.objfunmethod = 'numerical' 
                self.hexmethod = 'numerical' 
            else:
                self.objfunmethod = 'analytic' 
                self.hexmethod = 'analytic' 
            
    @property
    def liquidmodelname(self):
        return self._liquidmodelname
    
    @liquidmodelname.setter
    def liquidmodelname(self, new_liquidmodelname):
        self._liquidmodelname = new_liquidmodelname
        self.cbc_lmobject = lm.Context(lm.create(new_liquidmodelname))
        self.lletype = None
        
    @property       
    def mixturename(self):
        return self._mixturename
    
    @mixturename.setter
    def mixturename(self, new_mixturename):
        self._mixturename = new_mixturename
        self.cbc_lmobject.mixture = new_mixturename
        self.lletype = None
        
#     @property
#     def nc(self):
#         return self.cbc_lmobject.nc
        
    def set_bip(self, aij, bij, cij, dij, eij, fij):
        self.cbc_lmobject.set_bip(aij,bij,cij,dij,eij,fij)
        self.lletype = None
        
    def set_epsilon(self, epsilon12dk, epsilon11dk, epsilon22dk, Nc=4):
        self.epsilon12dk = epsilon12dk
        self.epsilon11dk = epsilon11dk
        self.epsilon22dk = epsilon22dk
        self.Nc = Nc
        self.cbc_lmobject.set_epsilon(epsilon12dk, epsilon11dk, epsilon22dk, Nc)
        self.lletype = None

    def set_cp(self, dbname, method):
        self.dbname = dbname
        self.method = method
        self.cbc_lmobject.set_cp(dbname, method)
    
    def __get_objfun(self):
        if self.objfunmethod == 'analytic':
            return self.cbc_lmobject.calc_dmudxa()[0,0], self.cbc_lmobject.calc_d2mudx2a()[0,0]
        elif self.objfunmethod == 'numerical':
            return self.cbc_lmobject.calc_dmudxn()[0,0], self.cbc_lmobject.calc_d2mudx2n()[0,0]
        elif self.objfunmethod is None:
            print('Please input the method name of objective function of critical point as analytic or numerical.')
        else:
            print('The method name of objective function of critical point should be analytic or numerical.')
            
    def __get_s(self):
        if self.hexmethod == 'analytic':
            return self.cbc_lmobject.calc_sa()
        elif self.hexmethod == 'numerical':
            return self.cbc_lmobject.calc_sn()
        elif self.hexmethod is None:
            print('Please input the excess enthalpy method name as analytic or numerical.')
        else:
            print('The method of excess enthalpy should be analytic or numerical.')
    
    def critical_objfun(self, xi):
        # Critical temperature & composition
        ti, x1i = xi
        x2i = 1 - x1i
        # print('i', xi)
        if (0. < ti < 1500.) and (0. <= x1i <= 1.):
            self.cbc_lmobject.set_tx(ti, np.array([x1i,x2i]))
            # The equations below are solved simultaneously to find the critical
            f1i, f2i = self.__get_objfun()
            # print('i', xi, f1i, f2i)
        elif (x1i <= 0) or (x1i >= 1.):
            f1i, f2i = ( 1 / x1i ) **2, ti **2
        else:
            f1i, f2i = x1i **2, ti **2
        return [f1i, f2i]
    
    def calc_critical_tx(self, objfunmethodi=None, lcp=False, tge=None, x1ge=None, solver='fsolve'):
        liquidmodelname = self._liquidmodelname
        if solver == 'least_squares':
            if liquidmodelname == 'COSMOSAC_MIT':
                bounds = ( [50., 1.e-5], [2000., 1-1.e-5] )
            else:
                bounds = ( [0., 1.e-5], [1500., 1-1.e-5] )
        if self.lletype is None:
            if liquidmodelname == 'COSMOSACb':
                tgi = [2.269] # Initial guess of temperature
                x1gi = [0.5] # Initial guess of composition
            elif liquidmodelname == 'COSMOSAC_MIT':
                tgi = [600.] # Initial guess of temperature
                x1gi = [0.5] # Initial guess of composition
            elif liquidmodelname[:8] == 'COSMOSAC':
                tgi = [50, 250, 400] # Initial guess of temperature
                x1gi = [0.15, 0.3, 0.8] # Initial guess of composition
            else:
                tgi = [120, 250, 400]
                x1gi = [0.15, 0.3, 0.8] # Initial guess of composition
            if type(tge) is list:
                tgi.extend(tge)
            elif type(tge) is float:
                tgi.append(tge)
            if type(x1ge) is list:
                x1gi.extend(x1ge)
            elif type(x1ge) is float:
                x1gi.append(x1ge)
            self.objfunmethod = objfunmethodi if objfunmethodi is not None else self.objfunmethod
            tci = np.array([])
            xc1i = np.array([])
            dmu1dx1ui = np.array([])
            dmu1dx1li = np.array([])
            objval = []
            deltat = 1 # increasing and decreasing temperature to judge UCP and LCP (K)
            tc_max = 1500. if liquidmodelname == 'COSMOSAC_MIT' else 800
            for tge in tgi:
                for x1ge in x1gi:
                    gi = [tge, x1ge]
                    if solver == 'fsolve':
                        sol, infodict, ier, mesg = fsolve(self.critical_objfun, gi, xtol=self.tol_critical[self.liquidmodelname],\
                                                          full_output=True)
                        objvale = infodict['fvec']
                    elif solver == 'least_squares':
                        resi = least_squares(fun=self.critical_objfun, x0=gi, bounds=bounds, xtol=self.tol_critical[self.liquidmodelname])
                        sol = resi.x.copy()
                        objvale = self.critical_objfun(sol)
                        status = resi.status
                        # print(sol, objvale, status)
                        if np.all(np.array(objvale) <= self.tol_critical[self.liquidmodelname]):
                            ier = 1
                        else:
                            ier = 0
                        
                    tce, xc1e = sol
                    xce = np.array([xc1e, 1 - xc1e])
                    
                    if ( ier == 1 ) and ( 0 <= xc1e <= 1 ) and ( min(tgi) < tce < tc_max ):
                        self.cbc_lmobject.set_tx(tce, xce)
                        dgmixe = self.cbc_lmobject.calc_dgmix()
                        self.cbc_lmobject.set_tx(tce + deltat, xce)
                        dmu1dx1ue, _ = self.__get_objfun()
                        self.cbc_lmobject.set_tx(tce - deltat, xce)
                        dmu1dx1le, _ = self.__get_objfun()
                        if ( ( dmu1dx1ue < 0 ) is not ( dmu1dx1le < 0 ) ) and ( dgmixe <= 0 ): 
                            # if sign of dmu1dx1u and dmu1dx1l are opposite and dgmix < 0 and T > 50 K
                            # print(tce, xc1e, dmu1dx1ue, dmu1dx1le)
                            tci = np.append(tci, tce)
                            xc1i = np.append(xc1i, xc1e)
                            dmu1dx1ui = np.append(dmu1dx1ui, dmu1dx1ue)
                            dmu1dx1li = np.append(dmu1dx1li, dmu1dx1le)
                    elif ( ier != 1 ) and ( 0 <= xc1e <= 1 ) and ( tce > 0 ):
                        objval.append(objvale)
            for tce in tci: # delete all elements that differences less than 1.e-5
                index = np.where(np.abs(tci - tce) < 1.)[0]
                tci = np.delete(tci, index[1:])
                xc1i = np.delete(xc1i, index[1:])
                dmu1dx1ui = np.delete(dmu1dx1ui, index[1:])
                dmu1dx1li = np.delete(dmu1dx1li, index[1:])
                del index
            if np.size(tci) == 0: # This mixture has no UCST
                print('This mixture has no UCST.')
                print(f'The minimum of dmu1/dx and d2mu1/dx12 = {objval}')
                self.tc = np.nan
                self.xc = np.full(self.cbc_lmobject.nc, np.nan)
                self.lletype = 'NCP'
                self.ncp = 0
            elif len(tci) == 1:
                tci = tci[0]
                xc1i = xc1i[0]
                xci = np.array([xc1i, 1 - xc1i])
                if lcp is True:
                    '''
                    self.cbc_lmobject.set_tx(tci + deltat, xci)
                    dmu1dx1i, _ = self.__get_objfun()
                    '''
                    dmu1dx1ue = dmu1dx1ui[0]
                    dmu1dx1le = dmu1dx1li[0]
                    if dmu1dx1ue > 0 and dmu1dx1le < 0:
                        """
                        tc + deltat => no phase separation 
                        tc - deltat => phase separation
                        """
                        self.lletype = 'UCP'
                        self.tc = {'UCP':tci}
                        self.xc = {'UCP':xci}
                    elif dmu1dx1ue < 0 and dmu1dx1le > 0: # LCP
                        """
                        tc + deltat => phase separation 
                        tc - deltat => no phase separation
                        """
                        self.lletype = 'LCP'
                        self.tc = {'LCP':tci}
                        self.xc = {'LCP':xci}
                    self.ncp = 1
                else:
                    self.tc = tci
                    self.xc = xci
                    self.lletype = 'UCP'
                self.cbc_lmobject.set_tx(self.tc, self.xc)
            else: # len(tci) >= 2
                '''
                tcu = max(tci)
                indexu = np.where(tci==tcu)[0][0]
                xcu = np.array([xc1i[indexu], 1 - xc1i[indexu]])
                tcl = min(tci)
                indexl = np.where(tci==tcl)[0][0]
                xcl = np.array([xc1i[indexl], 1 - xc1i[indexl]])
                '''
                if lcp is True:
                    lletype = []
                    self.tc = {}
                    self.xc = {}
                    nucp = 0 # number of UCP
                    nlcp = 0 # number of LCP
                    df_cp = pd.DataFrame()
                    df_cp['Tc'], df_cp['xc1'], df_cp['dmu1dx1u'], df_cp['dmu1dx1l'] = tci, xc1i, dmu1dx1ui, dmu1dx1li
                    index_cp = df_cp.index
                    df_cp.sort_values('Tc', inplace=True, ascending=False)
                    df_cp.index = index_cp
                    for (tce, xc1e, dmu1dx1ue, dmu1dx1le) in zip(df_cp['Tc'], df_cp['xc1'], df_cp['dmu1dx1u'], df_cp['dmu1dx1l']):
                        xce = np.array([xc1e, 1-xc1e])
                        if dmu1dx1ue > 0 and dmu1dx1le < 0:
                            """
                            tc + deltat => no phase separation 
                            tc - deltat => phase separation
                            """
                            nucp += 1
                            lletype.append('UCP')
                            if nucp == 1:
                                self.tc['UCP'] = tce
                                self.xc['UCP'] = xce
                            else:
                                self.tc['UCP'+str(nucp)] = tce
                                self.xc['UCP'+str(nucp)] = xce
                        elif dmu1dx1ue < 0 and dmu1dx1le > 0:
                            """
                            tc + deltat => phase separation 
                            tc - deltat => no phase separation
                            """
                            nlcp += 1
                            lletype.append('LCP')
                            if nlcp == 1:
                                self.tc['LCP'] = tce
                                self.xc['LCP'] = xce
                            else:
                                self.tc['LCP'+str(nlcp)] = tce
                                self.xc['LCP'+str(nlcp)] = xce
                    self.lletype = '+'.join(lletype)
                    self.ncp = nucp + nlcp
                    '''
                    self.cbc_lmobject.set_tx(tcu + 10, xcu)
                    dmu1dx1u, _ = self.__get_objfun()
                    self.cbc_lmobject.set_tx(tcl + 10, xcl)
                    dmu1dx1l, _ = self.__get_objfun()
                    if dmu1dx1u > 0 and dmu1dx1l < 0:
                        """
                        tcu + 10K => no phase separation 
                        tcl + 10K => phase separation
                        """
                        self.lletype = 'UCP+LCP'
                    elif dmu1dx1u < 0 and dmu1dx1l > 0:
                        """
                        tcu + 10K => phase separation 
                        tcl + 10K => no phase separation
                        """
                        self.lletype = 'LCP+UCP'
                    elif dmu1dx1u < 0 and dmu1dx1l < 0:
                        """
                        tcu + 10K => phase separation 
                        tcl + 10K => phase separation
                        """
                        self.lletype = 'LCP+LCP'
                        print(f'LCP+LCP occured dmu1dx1u = {dmu1dx1u} and dmu1dx1l= {dmu1dx1l}')
                    elif dmu1dx1u > 0 and dmu1dx1l > 0:
                        """
                        tcu + 10K => no phase separation 
                        tcl + 10K => no phase separation
                        """
                        self.lletype = 'UCP+UCP'
                        print(f'UCP+UCP occured dmu1dx1u = {dmu1dx1u} and dmu1dx1l= {dmu1dx1l}')
                    else:
                        print(f'dmu1dx1u = {dmu1dx1u} and dmu1dx1l= {dmu1dx1l}')
                    self.tc = {'UCP':tcu, 'LCP':tcl}
                    self.xc = {'UCP':xcu, 'LCP':xcl}
                    '''
                else:
                    tcu = max(tci)
                    indexu = np.where(tci==tcu)[0][0]
                    xcu = np.array([xc1i[indexu], 1 - xc1i[indexu]])
                    self.tc = tcu
                    self.xc = xcu
                    self.lletype = 'UCP'
        else:
            pass
        '''
        if self.flag_criticaldata is False:
            gi = [300, 0.5] # Initial guess
            fcorr = -1.1
            self.objfunmethod = objfunmethodi if objfunmethodi is not None else self.objfunmethod
            tci, xc1i = fsolve(self.critical_objfun, gi, xtol=self.tol_critical[self.liquidmodelname])
            sol, infodict, ier, mesg = fsolve(self.critical_objfun, gi, xtol=self.tol_critical[self.liquidmodelname], full_output=True)
            obj_value = infodict['fvec']
            if ( ier == 1 ):
                self.tc = sol[0]
                self.xc = np.array([sol[1], 1 - sol[1]])
                while np.any(self.xc < 0):
                    fcorr *= fcorr
                    gi[0] += 50*fcorr
                    sol, infodict, ier, mesg = fsolve(self.critical_objfun, gi, xtol=self.tol_critical[self.liquidmodelname]\
                                                      , full_output=True)
                    obj_value = infodict['fvec']
                    self.tc = sol[0]
                    self.xc = np.array([sol[1], 1 - sol[1]])
                    if gi[0] < 100 or gi[0] > 600:
                        self.tc = np.nan
                        self.xc = np.full(self.cbc_lmobject.nc, np.nan)
                        break
                self.flag_ucst = True
                self.cbc_lmobject.set_tx(self.tc, self.xc)
            else: # This mixture has no UCST
                print('This mixture has no UCST.')
                print(f'The minimum of dmu1/dx and d2mu1/dx12 = {obj_value}')
                self.tc = np.nan
                self.xc = np.full(self.cbc_lmobject.nc, np.nan)
                self.flag_ucst = False
            self.flag_criticaldata = True
        else:
            pass
        '''
    def calc_muc(self):
        self.calc_critical_tx()
        return self.cbc_lmobject.calc_mu()
    
    def calc_sc(self):
        self.calc_critical_tx()
        return self.__get_s()
    
    def calc_delta12c(self):
        self.calc_critical_tx()
        mui = self.cbc_lmobject.calc_mu()
        return mui[0] - mui[1]
        
'''        
class CalculationStrategy(metaclass=ABCMeta):
    
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """
'''
    
class Calc_lle:
    
    tol_lle_x = {'NRTL':1.e-12, 
                 'Wilson':1.e-12, 
                 'Margules':1.e-12,
                 'COSMOSAC2002e':1.e-9,
                 'COSMOSACb':1.e-12,
                 'COSMOSAC_MIT':1.e-7}
    
    tol_lle_f = {'NRTL':1.e-10, 
                 'Wilson':1.e-10, 
                 'Margules':1.e-10,
                 'COSMOSAC2002e':1.e-6,
                 'COSMOSACb':1.e-10,
                 'COSMOSAC_MIT':1.e-6}
    
    def __init__(self, liquidmodelname, mixturename=None):
        self._liquidmodelname = liquidmodelname
        self.clle_lmobject = lm.Context(lm.create(liquidmodelname))
        if mixturename is not None:
            self._mixturename = mixturename
            self.clle_lmobject.mixture = mixturename
        
    def set_bip(self, aij, bij, cij, dij, eij, fij):
        self.clle_lmobject.set_bip(aij, bij, cij, dij, eij, fij)
        
    def set_epsilon(self, epsilon12dk, epsilon11dk, epsilon22dk, Nc=4):
        self.epsilon12dk = epsilon12dk
        self.epsilon11dk = epsilon11dk
        self.epsilon22dk = epsilon22dk
        self.Nc = Nc
        self.clle_lmobject.set_epsilon(epsilon12dk, epsilon11dk, epsilon22dk, Nc)

    def set_cp(self, dbname, method):
        self.dbname = dbname
        self.method = method
        self.clle_lmobject.set_cp(dbname, method)
        
    @property
    def liquidmodelname(self):
        return self._liquidmodelname
    
    @liquidmodelname.setter
    def liquidmodelname(self, new_liquidmodelname):
        self._liquidmodelname = new_liquidmodelname
        self.clle_lmobject = lm.Context(lm.create(new_liquidmodelname))
        
    @property       
    def mixturename(self):
        return self._mixturename
    
    @mixturename.setter
    def mixturename(self, new_mixturename):
        self._mixturename = new_mixturename
        self.clle_lmobject.mixture = new_mixturename
        
    @property
    def nc(self):
        return self.clle_lmobject.nc
        
    def check_x(self, xa, xb):
        tol_lle_f = self.tol_lle_f[self.liquidmodelname]
        tol_lle_xaxb = self.tol_lle_f[self.liquidmodelname] * 1000
        f_reasonable_x = True
        # check whether xi not nan in all phases
        if np.any(np.isnan(xa)) or np.any(np.isnan(xb)):
            f_reasonable_x = False
        # check whether x1+x2+...+xc = 1 in all phases
        if ( ( np.sum(xa) - 1 ) >= tol_lle_f ) or ( ( np.sum(xb) - 1 ) >= tol_lle_f ):
            f_reasonable_x = False
        # check whether 0 <= xi <=1 for all i in all phases
        if not ( np.all(0 <= xa) and np.all(xa <= 1) ) or not ( np.all(0 <= xb) and np.all(xb <= 1) ):
            f_reasonable_x = False
        # check whether all xai != xbi
        if ( np.all(np.abs(xa - xb) <= tol_lle_xaxb) ):
            f_reasonable_x = False
        return f_reasonable_x
        
    def ntsol_mb(self, Ki): #用於zi和Ki為1維矩陣(不同成分)
        zi = self.z
        lfai = 0.5 # initial guess of liquid fraction of phase a
        tol_lle_x = self.tol_lle_x[self.liquidmodelname]
        while True:
            xai = zi / ( lfai + Ki * ( 1 - lfai ) ) # a:water-rich, b:n-butanol-rich
            xbi = xai * Ki
            F = abs( sum( xbi ) - sum( xai ) )
            if F >= tol_lle_x :
                dFdlfai = - sum( zi * ( Ki - 1 ) * ( 1 - Ki * lfai ) / ( lfai + Ki * ( 1 - lfai ) ) **2 )
                lfai -= ( sum( xbi ) - sum( xai ) ) / dFdlfai
            else:
                break
        return lfai, xai, xbi
    
    def isothermal_flash(self, t, z=None, Kgi=None, imax=None, print_ef=False):
        self.t = t # temperature (K)
        self.z = z # feed composition
        ti = self.t
        zi = self.z
        nci = self.nc
        tol_f = self.tol_lle_f[self.liquidmodelname]
        Ki = np.logspace(-3, 2, nci) if ( Kgi is None ) else Kgi.copy() # Kgi: guess value of Ki(=xbgi/xagi)
        i = 0 # iteration times
        while True: 
            if nci == 2:
                xai = np.array([( 1 - Ki[1] ) / ( Ki[0] - Ki[1] ), ( Ki[0] - 1 ) / ( Ki[0] - Ki[1] )])
                xbi = xai * Ki
                lfai = ( zi[0] - xbi[0] ) / ( xai[0] - xbi[0] ) if zi is not None else None
            else:
                lfai, xai, xbi = self.ntsol_mb(Ki)
            self.clle_lmobject.set_tx(ti, xai)
            gammaai = self.clle_lmobject.calc_gammam()
            self.clle_lmobject.set_tx(ti, xbi)
            gammabi = self.clle_lmobject.calc_gammam()
            aai = xai * gammaai
            abi = xbi * gammabi
            ef = abs( abi / aai - 1 )
            if print_ef is True:
                print('error=', ef)
            if ( imax is not None ) and ( type(imax) is int ):
                if np.any(ef >= tol_f) and i <= imax:
                    Ki *= ( aai / abi )
                    i += 1
                elif np.any(ef >= tol_f) and i > imax: # error >= tolerance and i > imax (not converge)
                    xai = np.full(nci, np.nan)
                    xbi = np.full(nci, np.nan)
                    break
                else: # error < tolerance
                    break
            else: # imax is None or type(imax) is not int
                if ( imax is not None ) and ( type(imax) is not int ):
                    print(f'imax is not integer, it is {type(imax)}')
                if np.any(ef >= tol_f):
                    Ki *= ( aai / abi )
                    i += 1
                else:
                    break
        self.f_reasonable_x = self.check_x(xai, xbi)
        # print(ef)
        '''
        if not ( ( 0 <= xai[0] <= 1 ) and ( 0 <= xbi[0] <= 1 ) ):
            Ki = np.logspace(-4, 3, nci)
            while True: #判斷所有成分的error中否有任何一個大於10的-6次方的
                if nci == 2:
                    xai = np.array([( 1 - Ki[1] ) / ( Ki[0] - Ki[1] ), ( Ki[0] - 1 ) / ( Ki[0] - Ki[1] )])
                    xbi = xai * Ki
                    lfai = ( zi[0] - xbi[0] ) / ( xai[0] - xbi[0] ) if zi is not None else None
                else:
                    lfai, xai, xbi = self.ntsol_mb(Ki)
                self.clle_lmobject.set_tx(ti, xai)
                gammaai = self.clle_lmobject.calc_gammam()
                self.clle_lmobject.set_tx(ti, xbi)
                gammabi = self.clle_lmobject.calc_gammam()
                aai = xai * gammaai
                abi = xbi * gammabi
                ef = abs( abi / aai - 1 )
                if np.any(ef >= tol_f):
                    Ki *= ( aai / abi )
                else:
                    break
        '''
        return lfai, xai, xbi
    

class Calc_binary_nonclassical(Calc_binary_critical):
    
    tol_tdelta12_x = {'NRTL':1.e-10,
                      'Wilson':1.e-10,
                      'COSMOSAC2002e':1.e-8}
    tol_xp = {'NRTL':1.e-9, 
              'Wilson':1.e-9,
              'COSMOSAC2002e':1.e-7}
    tol_xp_depablo = {'NRTL':1.e-9, 
                      'Wilson':1.e-9, 
                      'COSMOSAC2002e':1.e-7}
    ddelta12p_depablo = {'NRTL':1.e-7, 
                         'COSMOSAC2002e':1.e-6}
    tol_tp = {'NRTL':1.e-11, 
              'Wilson':1.e-11, 
              'COSMOSAC2002e':1.e-9}
    
    def __init__(self, alphai, betai, lambdai, wi, liquidmodelname, mixturename=None, hexmethod=None, objfunmethod=None):
        super().__init__(liquidmodelname, mixturename, objfunmethod, hexmethod)
        self.alphai = alphai
        self.betai = betai
        self.lambdai = lambdai
        self.wi = wi
        self.cbn_lmobject = lm.Context(lm.create(liquidmodelname))
        self.cbn_lmobject.mixture = mixturename
        self.cbn_lleobject = Calc_lle(liquidmodelname, mixturename)

    def set_tx(self, ti, xi):
        self.t = ti
        self.x = xi
    
    def __check_tx(self):
        if self.t is None and self.x is None:
            print('Please set the temperature and composition.')
        elif self.t is None:
            print('Please set the temperture.')
        elif self.x is None:
            print('Please set the composition.')
        else:
            pass
        
    def set_bip(self, aij, bij, cij, dij, eij, fij):
        super().set_bip(aij, bij, cij, dij, eij, fij)
        self.cbn_lmobject.set_bip(aij, bij, cij, dij, eij, fij)
        
    def calc_criticaldata(self):
        self.calc_critical_tx()
        self.muc = self.calc_muc()
        self.sc = self.calc_sc()
        self.delta12c = self.calc_delta12c()
        
    @property
    def mu(self):
        try:
            self.cbn_lmobject.set_tx(self.t, self.x)
            return self.cbn_lmobject.calc_mu()
        except:
            self.__check_tx()
        
    @property
    def s(self):
        try:
            self.cbn_lmobject.set_tx(self.t, self.x)
            return self.__get_s()
        except:
            self.__check_tx()
            
    @property
    def delta12(self):
        try:
            self.cbn_lmobject.set_tx(self.t, self.x)
            mui = self.cbn_lmobject.calc_mu()
            return mui[0] - mui[1]
        except:
            self.__check_tx()
    
    @property
    def psib(self):
        self.calc_criticaldata()
        if self.lletype is not None:
            if abs( self.t - self.tc ) / self.tc < 1.e-12 and np.all( abs( self.x - self.xc ) / self.xc < 1.e-12 ):
                psibi = 0 
            else:
                tc_ti = self.tc / self.t
                psibi = -self.mu[1] + self.muc[1] * tc_ti - self.xc[0] * ( self.delta12 - self.delta12c * tc_ti ) - self.sc * ( 1 - tc_ti )
                psibi = 0 if psibi < 0 or abs(psibi) < 1.e-12 else psibi
        elif self.lletype == 'NCP':
            psibi = np.nan
        elif self.lletype is None:
            self.calc_criticaldata()
        else:
            print(f'The LLE type is not None, NCP, UCP or LCP, it is {self.lletype}')
            psibi = np.nan
        return psibi
 
    @property
    def psibs(self):
        return abs( self.psib / ( self.muc[1] * self.tc / self.t ) )
    
    @property
    def g_dpf(self):
        psibsl = self.psibs **( self.lambdai / 4 )
        return ( psibsl / ( self.wi + psibsl ) ) **( 1 / self.lambdai )
    
    @property
    def theta(self):
        return 2 * self.alphai / ( 2 - self.alphai )
    
    @property
    def phi(self):
        return 1 - 2 * self.betai * ( 1 + self.alphai / ( 2 - self.alphai ) )
    
    @property
    def tp(self):
        return self.tc + ( self.t - self.tc ) * self.g_dpf **self.theta
    
    @property
    def delta12xct(self):
        self.cbn_lmobject.set_tx(self.t, self.xc)
        muxcti = self.cbn_lmobject.calc_mu()
        self.cbn_lmobject.set_tx(self.t, self.x)
        return muxcti[0] - muxcti[1]
    
    @property
    def delta12xctp(self):
        self.cbn_lmobject.set_tx(self.tp, self.xc)
        muxctpi = self.cbn_lmobject.calc_mu()
        self.cbn_lmobject.set_tx(self.t, self.x)
        return muxctpi[0] - muxctpi[1]
        
    @property
    def delta12p(self):
        return self.delta12xctp + self.t / self.tp * ( self.delta12 - self.delta12xct ) * self.g_dpf **self.phi
    
    @property
    def mu2p(self):
        t_tpi = self.t / self.tp
        return - ( - self.mu[1] * t_tpi + self.xc[0] * ( self.delta12p - self.delta12 * t_tpi ) + self.sc * ( 1 - t_tpi ) )
    
    @property
    def mup(self):
        return np.array([self.delta12p + self.mu2p, self.mu2p])

    @property
    def xp(self):
        if self.psib == 0:
            xpi = self.xc
        else:
            ti, xi = self.t, self.x
            tpi = self.tp
            ddelta12p = 1.e-7
            delta12p_plus_minusi = (self.delta12p + ddelta12p, self.delta12p - ddelta12p)
            dmu2p_plus_minusi = []
            for delta12pi in delta12p_plus_minusi:
                def mu2p_plus_minus_objfun(xii):
                    tii = xii[0]
                    x1ii = xii[1]
                    x2ii = 1 - x1ii
                    self.set_tx(tii, np.array([x1ii, x2ii]))
                    ftpi = ( tpi - self.tp ) / tpi
                    fdelta12pi = ( delta12pi - self.delta12p ) / delta12pi
                    return [ftpi, fdelta12pi]
                soli = fsolve(mu2p_plus_minus_objfun, [ti, xi[0]], xtol=self.tol_xp[self.liquidmodelname])
                t_plus_minusi = soli[0]
                x_plus_minusi = np.array([soli[1], 1-soli[1]])
                self.set_tx(t_plus_minusi, x_plus_minusi)
                dmu2p_plus_minusi.append(self.mu2p)
            x1pi = - ( dmu2p_plus_minusi[0] - dmu2p_plus_minusi[1] ) / ( 2 * ddelta12p )
            xpi = np.array([x1pi, 1 - x1pi])
            self.set_tx(ti, xi) 
        return xpi
    
    def tdelta12_solve_x(self, ti, delta12i):
        toi = self.t
        xoi = self.x
        def tdelta12_solve_x_objfun(x1ii):
            self.set_tx(ti, np.array([x1ii[0], 1-x1ii[0]]))
            fdelta12i = ( delta12i - self.delta12 ) / delta12i
            return fdelta12i
        x1i = fsolve(tdelta12_solve_x_objfun, xoi[0], xtol=self.tol_tdelta12_x[self.liquidmodelname])[0]
        xi = np.array([x1i, 1-x1i])
        self.set_tx(toi, xoi)
        return xi
    
    @property
    def xp_depablo(self):
        if self.psib == 0:
            xpi = self.xc
        else:
            toi, xoi = self.t, self.x
            ti, xi = self.t, self.x
            mu2i = self.mu[1]
            tpi = self.tp
            ddelta12p = self.ddelta12p_depablo[self.liquidmodelname]
            delta12p = self.delta12p
            delta12p_plus_minusi = ( delta12p + ddelta12p, delta12p - ddelta12p )
            mu2p_plus_minusi = []
            for delta12pi in delta12p_plus_minusi:
                while True:
                    self.set_tx(ti, xi)
                    dpfi = self.g_dpf
                    tni = ( tpi - self.tc ) / ( dpfi **self.theta ) + self.tc
                    self.set_tx(tpi, self.xc)
                    delta12xctpi = self.delta12
                    self.set_tx(tni, self.xc)
                    delta12xctni = self.delta12
                    delta12ni = delta12xctni + tpi / ( tni * dpfi **self.phi ) * ( delta12pi - delta12xctpi )
                    self.set_tx(toi, xoi)
                    xni = self.tdelta12_solve_x(tni, delta12ni)
                    self.set_tx(tni, xni)
                    mu2ni = self.mu[1]
                    if abs( ( mu2i - mu2ni ) / mu2i ) > self.tol_xp_depablo[self.liquidmodelname]:
                        mu2i = mu2ni
                        ti = tni
                        xi = xni
                    else:
                        tn_tpi = tni / tpi
                        mu2pni = - ( - mu2ni * tn_tpi + self.xc[0] * ( delta12pi - delta12ni * tn_tpi ) + self.sc * ( 1 - tn_tpi ) )
                        mu2p_plus_minusi.append(mu2pni)
                        self.set_tx(toi, xoi)
                        break
            x1pi = - ( mu2p_plus_minusi[0] - mu2p_plus_minusi[1] ) / ( 2 * ddelta12p )
            xpi = np.array([x1pi, 1 - x1pi])
        return xpi

    def texp_solve_t(self, texpi):
        if self.lletype is not None and self.lletype != 'NCP': # critical data have been calculated and this mixture has UCST
            if texpi < self.tc:
                ti = texpi
                texp = texpi
                while True:
                    toi, xoi = self.t, self.x
                    lfai, xai, xbi = self.cbn_lleobject.isothermal_flash(ti)
                    self.set_tx(ti, xai)
                    tpn = self.tp
                    et = ( texp - tpn ) / texp
                    if abs(et) >= self.tol_tp[self.liquidmodelname]:
                        ti +=  et * texp
                        self.set_tx(toi, xoi)
                    else:
                        self.set_tx(toi, xoi)
                        break
            elif texpi == self.tc:
                ti = self.tc
                xai = xbi = self.xc
            elif texpi > self.tc:
                ti = np.nan
                xai = xbi = np.full(self.cbn_lleobject.nc, np.nan)
                print('Can not separate to two phase. (T>Tc)')
            else:
                print(f'texp is not float or integer, it is {type(texp)}')
            return ti, xai, xbi
        elif self.self.lletype == 'NCP': # critical data have been calculated and this mixture has no UCST
            ti = np.nan
            xai = xbi = np.full(self.cbn_lleobject.nc, np.nan)
            return ti, xai, xbi
        elif self.self.lletype is None: # critical data have not been calculated
            self.calc_criticaldata()
        else:
            print(f'The LLE type = {self.lletype}, Tc = {self.tc}')
          
        
        
def decorator_d2mdt2n(mi):
    """ 
    A decorator for numerically calculate second derivative of temperature of properties. 
    """
    def method(mth):
        def wrapper(self, dt):
            """
            Parameters
            ----------
            dti : float
                step size of temperature for the numerical differentiation
            ti : float
                temperture (K) 
            mi : function
                any property that is function of temperture
            mth : regular method
                the regular method you want to decorate it.
            mci : float or numpy array, float nX1 matrix
                property in the state of ti
            mplusi : float or numpy array, float nX1 matrix
                property in the state of ti + dti
            mminusi : float or numpy array, float nX1 matrix
                property in the state of ti - dti
            d2mdt2ni : float or numpy array, float nXn matrix
                second derivatives of the property (numerical equation)

            """
            dti = dt
            ti = self.t
            mci = mi(self)
            self.set_t(ti + dti)
            mplusi = mi(self)
            self.set_t(ti - dti)
            mminusi = mi(self)
            d2mdt2ni = ( mplusi - 2 * mci + mminusi ) / ( dti **2 )
            self.set_t(ti)
            return d2mdt2ni
        return wrapper
    return method
        
        
class Calc_binary_rg:
    # LLE calculation by coarse-graining RG
    tol_lle_dgmix = {'NRTL':1.e-12,
                     'Wilson':1.e-12,
                     'Margules':1.e-12,
                     'COSMOSAC2002e':1.e-9,
                     'COSMOSACb':1.e-12,
                     'COSMOSAC_MIT':1.e-7}
    
    def __init__(self, liquidmodelname, mixturename=None, lhat0=None, delta=None, deltal=1.\
                 , iteration=None, iterationdata=False, nx=1001, d=3, c_rg=2., dt=1.e-4, rgtype=None):
        """
        Parameters
        ----------
        liquidmodelname : string
            The name of liquid model
        mixturename : string
            The name of this mixture
        lhat0i : float
            one of Renormalization Group parameter 
        deltai : float
            one of Renormalization Group parameter 
        iteration : int or None
            The maximum iteration number including mean-field free energy, dGmix0
        iterationi : int or None
            initial setting of total iteration time you want to calculate
        iterationdata : boolean
            Whether need to leave iteration data
        nx : int
            total number of mole fraction data
        nc : int
            The Number of components in this mixture
        d : int
            The number of dimension
        c_rg : int
            The factor by which the length of a cell is increased per iteration
        dt : float 
            differential temperature, used in the calculation of heat capacity
        dgmix_rg : None or numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            total molar Gibbs free energy of mixing divide by RT for all mole fraction data (calculated by liquid modol+RG)
        ----------
        object
        ----------
        cblleperg_lmobject : 
            The object of class "Context" in module "liquidmodel"
            
        """
        self._liquidmodelname = liquidmodelname
        self.lhat0 = lhat0
        self.delta = delta
        self.deltal = deltal
        self.iteration = iteration
        self.iterationi = iteration # initial iteration
        self.iterationdata = iterationdata
        self._nx = nx
        self.d = d
        self.c_rg = c_rg
        self.dt = dt
        self.rgtype = rgtype
        self.cbllerg_lmobject = lm.Context(lm.create(liquidmodelname))
        if mixturename is not None:
            self._mixturename = mixturename 
            self.cbllerg_lmobject.mixture = mixturename
        self.__dgmix_rg = None
        self.__dgmix_rgaw = None
        #self.z = 2 #coordinate number

    @property
    def liquidmodelname(self):
        return self._liquidmodelname
    
    @liquidmodelname.setter
    def liquidmodelname(self, new_liquidmodelname):
        self._liquidmodelname = new_liquidmodelname
        self.cbllerg_lmobject = lm.Context(lm.create(new_liquidmodelname))
        self.iteration = self.iterationi
        self.__dgmix_rg = None
        self.__dgmix_rgaw = None
        
    @property       
    def mixturename(self):
        return self._mixturename
    
    @mixturename.setter
    def mixturename(self, new_mixturename):
        self._mixturename = new_mixturename
        self.cbllerg_lmobject.mixture = new_mixturename
        self.iteration = self.iterationi
        self.__dgmix_rg = None
        self.__dgmix_rgaw = None
        
    @property
    def nc(self):
        return self.cbllerg_lmobject.nc
        
    @property       
    def nx(self):
        return self._nx
    
    @nx.setter
    def nx(self, new_nx):
        self._nx = new_nx
        self.iteration = self.iterationi
        self.__dgmix_rg = None
        self.__dgmix_rgaw = None
        
    def set_bip(self, aij, bij, cij, dij, eij, fij):
        self.cbllerg_lmobject.set_bip(aij, bij, cij, dij, eij, fij)
        self.iteration = self.iterationi
        self.__dgmix_rg = None
        self.__dgmix_rgaw = None
        
    def set_epsilon(self, epsilon12dk, epsilon11dk, epsilon22dk, Nc=4, f_bip=True):
        self.epsilon12dk = epsilon12dk
        self.epsilon11dk = epsilon11dk
        self.epsilon22dk = epsilon22dk
        self.Nc = Nc
        if f_bip is True:
            self.cbllerg_lmobject.set_epsilon(epsilon12dk, epsilon11dk, epsilon22dk, Nc)
        self.__dgmix_rg = None
        self.__dgmix_rgaw = None

    def set_cp(self, dbname, method):
        self.dbname = dbname
        self.method = method
        self.cbllerg_lmobject.set_cp(dbname, method)
    
    def set_rgp(self, lhat0=None, delta=None, deltal=None, iteration=None, iterationdata=None, rgtype=None):
        if lhat0 is not None:
            self.lhat0 = lhat0
        if delta is not None:
            self.delta = delta
        if deltal is not None:
            self.deltal = deltal
        if iteration is not None:
            self.iteration = iteration
            self.iterationi = iteration
        if iterationdata is not None:
            self.iterationdata = iterationdata
        if rgtype is not None:
            self.rgtype = rgtype
        self.__dgmix_rg = None
        self.__dgmix_rgaw = None
            
    def set_t(self, ti):
        self.t = ti
        self.iteration = self.iterationi
        self.__dgmix_rg = None
        self.__dgmix_rgaw = None
    
    @property
    def nymax(self):
        return int( ( self.nx - 1 ) / 2 + 1 )
    
    @property
    def jx_tail(self):
        return np.array(self.xmin * ( self.nx - 1 ) + 1, dtype = int) # x: 0 -> 1, iy: 1 -> 501 -> 1

    @property
    def jmx_tail(self):
        return np.arange(self.nx) + 1  # x: 0 -> 1, jpx_tail: 1 -> 1001
        
    @property
    def jpx_tail(self):
        return self.nx - np.arange(self.nx)  # x: 0 -> 1, jpx_tail: 1001 -> 1

    @property
    def jpx_end(self):
        return self.jpx_tail - 1  # x: 0 -> 1, jpx_tail: 1000 -> 0
    
    @property
    def dx(self):
        return 1 / ( self.nx - 1 )
    
    @property
    def x(self):
        x1i = np.linspace(0, 1, self.nx)
        return np.array([x1i, np.flipud(x1i)])
    
    @property
    def xmin(self):
        return np.amin(self.x, axis=0) # x: 0 -> 1, xmin: 0 -> 0.5 -> 0
    
    @property
    def ymax(self):
        return self.xmin[:self.nymax] # ymax: 0 -> 0.5
    
    @property
    def nmax(self): 
        """ The maximum number of iteration due to the limitation of maximum number of floating point"""
        maxnum = np.finfo(float).max * ( 1 - 1e-13 ) # the maximum number of floating point
        return int(np.floor(np.emath.logn(self.c_rg, maxnum **( 1 / self.d ) / self.lhat0)))
    
    @property
    def u(self):
        ti = self.t
        flag_setx = hasattr(self.cbllerg_lmobject, 'x')
        if flag_setx:
            xoi = self.cbllerg_lmobject.x
        self.cbllerg_lmobject.set_tx(ti, np.array([0.5,0.5]))
        ui = self.cbllerg_lmobject.u()
        if flag_setx:
            self.cbllerg_lmobject.set_tx(ti, xoi)
        '''
        if self.liquidmodelname[:5] == 'COSMO':
            ti = self.t
            self.cbllerg_lmobject.set_tx(ti, np.array([0.5,0.5]))
            ui = self.cbllerg_lmobject.u()
        else:
            zi = self.z
            ti = self.t
            self.cbllerg_lmobject.set_tx(ti, np.array([0.5,0.5]))
            ui = self.cbllerg_lmobject.u() * zi / 2
            #ui = self.cbllerg_lmobject.calc_gex()
        '''
        return ui
    
    @property
    def dgmix_cg(self):
        ti = self.t
        xi = self.x
        dgmixi = np.zeros(self.nx, dtype=float)
        for ix, xe in enumerate(xi.T):
            self.cbllerg_lmobject.set_tx(ti, xe)
            dgmixi[ix] = self.cbllerg_lmobject.calc_dgmix()
        return dgmixi
    
    @property
    def ddgmixdx1_cg(self):
        """
        Parameters
        ----------
        ddgmixdx1_cg : numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            first derivatives of Gibbs free energy divide by RT (numerical equation)
        """
        xi = self.x
        dxi = self.dx
        dgmix_cg = self.dgmix_cg
        ddgmixdx1_cg = np.zeros(self.nx, dtype=float)
        for ix, xe in enumerate(xi.T):
            if ix == 0:
                ddgmixdx1_cg[ix] = ( dgmix_cg[ix+1] - dgmix_cg[ix] ) / dxi
            elif ( ix == self.nx - 1 ):
                ddgmixdx1_cg[ix] = ( dgmix_cg[ix] - dgmix_cg[ix-1] ) / dxi
            else:
                ddgmixdx1_cg[ix] = ( dgmix_cg[ix+1] - dgmix_cg[ix-1] ) / ( 2 * dxi )
        return ddgmixdx1_cg

    @property
    def d2dgmixdx12_cg(self):
        """
        Parameters
        ----------
        ddgmixdx1_cg : numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            first derivatives of Gibbs free energy divide by RT (numerical equation)
        """
        xi = self.x
        dxi = self.dx
        dgmix_cg = self.dgmix_cg
        d2dgmixdx12_cg = np.zeros(self.nx, dtype=float)
        for ix, xe in enumerate(xi.T):
            if ix == 0:
                d2dgmixdx12_cg[ix] = ( dgmix_cg[ix+2] - 2 * dgmix_cg[ix+1] + dgmix_cg[ix] ) / ( dxi **2 )
            elif ( ix == self.nx - 1 ):
                d2dgmixdx12_cg[ix] = ( dgmix_cg[ix-2] - 2 * dgmix_cg[ix-1] + dgmix_cg[ix] ) / ( dxi **2 )
            else:
                d2dgmixdx12_cg[ix] = ( dgmix_cg[ix+1] - 2 * dgmix_cg[ix] + dgmix_cg[ix-1] ) / ( dxi **2 )
        return d2dgmixdx12_cg
    
    def calc_dgmix_rg(self):
        """
        Parameters
        ----------
        xi : numpy array, float ncXnx
            mole fraction of component i  
        dxi : float
            differential mole fraction of component i  
        lhat0i : float
            one of Renormalization Group parameter 
        deltai : float
            one of Renormalization Group parameter 
        iteration : int or None
            total iteration time you want to calculate
        iterationdata : boolean
            Whether need to leave iteration data
        nx : int
            total number of mole fraction data
        nymax : int
            total number of mole fraction data for maximum varible y
        nid : int
            actual iteration data for gibbs free energy of mixing 
            (if iterationdata is True: iteration, else: 1)
        xmin : numpy array, float 1Xnx
            minimum mole fraction for all component in xi
            eg. if x[i] = np.array([0.3, 0.7]), xmin[i] = 0.3
        ymax : numpy array, float 1Xnymax
            minimum mole fraction for all component in xi
        ui : float
            one of Renormalization Group parameter (total mixing internal energy for x1=0.5)
        tol_dgmix : float
            the tolerance for Renormalization Group iteration 
        n : int
            iteration step
        d : int
            dimension
        dgmix0 : numpy array,.float 1Xnx
            total molar Gibbs free energy of mixing divide by RT for all mole fraction data
        dgmixn : numpy array, float 1Xnx
            total molar Gibbs free energy of mixing divide by RT for all mole fraction data at nth iteration
        dgmixnm1 : numpy array, float 1Xnx
            total molar Gibbs free energy of mixing divide by RT for all mole fraction data at (n-1)th iteration
        dgmix_rg : None or numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            total molar Gibbs free energy of mixing divide by RT for all mole fraction data (calculated by liquid modol+RG)
        deltagn : numpy array, float 1Xnx
            correction to total molar Gibbs free energy of mixing divide by RT for all mole fraction data at nth iteration
        deltag_rg : numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            correction to total molar Gibbs free energy of mixing divide by RT for all mole fraction data 
            (calculated by liquid modol+RG)
            
        """
        
        if self.__dgmix_rg is not None: # dgmix_rg have been calculated
            return self.__dgmix_rg
        """Get class attributes"""
        xi = self.x
        dxi = self.dx
        lhat0i = self.lhat0
        deltai = self.delta
        deltali = self.deltal
        iteration = self.iteration
        iterationdata = self.iterationdata
        nx = self.nx
        nymax = self.nymax
        jx_tail = self.jx_tail
        xmin = self.xmin
        ymax = self.ymax
        dgmix0 = self.dgmix_cg
        d = self.d
        ui = self.u  
        c_rg = self.c_rg
        tol_dgmix = self.tol_lle_dgmix[self.liquidmodelname]

        """Declare array"""
        gsn = np.zeros((nx, nymax), dtype=float)
        gln = np.zeros((nx, nymax), dtype=float)
        overflowfactorsn = np.zeros(nx, dtype=float)
        overflowfactorln = np.zeros(nx, dtype=float)
        numeratorn = np.zeros(nx, dtype=float)
        denominatorn = np.zeros(nx, dtype=float)

        if iterationdata is True:
            gs_rg = np.zeros((1, nx, nymax), dtype=float)
            gl_rg = np.zeros((1, nx, nymax), dtype=float)
            numerator_rg = np.zeros((1, nx), dtype=float)
            denominator_rg = np.zeros((1, nx), dtype=float)
            deltag_rg = np.zeros((1, nx), dtype=float)
            dgmix_rg = dgmix0.copy().reshape(1, nx)
        
        n = 0 
        dgmixn = dgmix0.copy()
        while True:
            n += 1
            vhatn = ( c_rg **n * lhat0i ) **d
            dgmixnm1 = dgmixn.copy()
            for ix, xe in enumerate(xi.T):
                for jx, ye in enumerate(ymax[:jx_tail[ix]]):
                    flucl = ui * ye **2 * deltali
                    flucs = c_rg **( - 2 * n ) * flucl * deltai
                    gsln = ( dgmixnm1[ix+jx] + dgmixnm1[ix-jx] ) / 2 - dgmixnm1[ix]
                    gsn[ix,jx] = gsln + flucs
                    gln[ix,jx] = gsln + flucl
                overflowfactorsn[ix] = np.max( - gsn[ix,:] )
                overflowfactorln[ix] = np.max( - gln[ix,:] )
                numeratorn[ix] = ( np.sum( np.exp( ( - gsn[ix,1:jx] - overflowfactorsn[ix] ) * vhatn ) ) + \
                                   ( np.exp( ( - gsn[ix,0] - overflowfactorsn[ix] ) * vhatn ) + \
                                    np.exp( ( - gsn[ix,jx] - overflowfactorsn[ix] ) * vhatn ) ) / 2 ) * dxi
                denominatorn[ix] = ( np.sum( np.exp( ( - gln[ix,1:jx] - overflowfactorln[ix] ) * vhatn ) ) + \
                                     ( np.exp( ( - gln[ix,0] - overflowfactorln[ix] ) * vhatn ) + \
                                      np.exp( ( - gln[ix,jx] - overflowfactorln[ix] ) * vhatn ) ) / 2 ) * dxi

                '''
                 *  Add a term in the exponential (each amplitude is weighted by exp(-y))
                numeratorn[ix] = ( np.sum( np.exp( ( - gsn[ix,1:jx] - overflowfactorsn[ix] ) * vhatn + ymax[1:jx] ) ) + \
                                   ( np.exp( ( - gsn[ix,0] - overflowfactorsn[ix] ) * vhatn + ymax[0] ) + \
                                    np.exp( ( - gsn[ix,jx] - overflowfactorsn[ix] ) * vhatn + ymax[jx] ) ) / 2 ) * dxi
                denominatorn[ix] = ( np.sum( np.exp( ( - gln[ix,1:jx] - overflowfactorln[ix] ) * vhatn + ymax[1:jx] ) ) + \
                                     ( np.exp( ( - gln[ix,0] - overflowfactorln[ix] ) * vhatn + ymax[0] ) + \
                                      np.exp( ( - gln[ix,jx] - overflowfactorln[ix] ) * vhatn + ymax[jx] ) ) / 2 ) * dxi

                --------------------------------
                 * Simpson's method
                if (jx_tail[ix] <= 2):
                    numeratorn[ix] = ( np.sum( np.exp( ( - gsn[ix,1:jx] - overflowfactorsn[ix] ) * vhatn ) ) + \
                                       ( np.exp( ( - gsn[ix,0] - overflowfactorsn[ix] ) * vhatn ) + \
                                        np.exp( ( - gsn[ix,jx] - overflowfactorsn[ix] ) * vhatn ) ) / 2 ) * dxi
                    denominatorn[ix] = ( np.sum( np.exp( ( - gln[ix,1:jx] - overflowfactorln[ix] ) * vhatn ) ) + \
                                         ( np.exp( ( - gln[ix,0] - overflowfactorln[ix] ) * vhatn ) + \
                                          np.exp( ( - gln[ix,jx] - overflowfactorln[ix] ) * vhatn ) ) / 2 ) * dxi
                else:
                    funcs = np.exp( ( - gsn[ix,:jx+1] - overflowfactorsn[ix] ) * vhatn )
                    funcl = np.exp( ( - gln[ix,:jx+1] - overflowfactorln[ix] ) * vhatn )
                    numeratorn[ix] = simpson(y=funcs,  dx=dxi)
                    denominatorn[ix] = simpson(y=funcl,  dx=dxi)
                '''
            deltagn = - np.log( numeratorn / denominatorn ) / vhatn - ( overflowfactorsn - overflowfactorln )
            # print(f'numeratorn={numeratorn[501]}, denominatorn={denominatorn[501]}, vhatn={vhatn}, overflowfactorsn={overflowfactorsn}, overflowfactorln={overflowfactorln}')
            dgmixn = dgmixnm1 + deltagn
            egn = np.max(np.abs(deltagn[1:-1]))
            # egn = np.max(np.abs(deltagn[1:-1] / dgmixn[1:-1]))
            if iterationdata is True: 
                gs_rg = np.append(gs_rg, gsn.reshape(1, nx, nymax), axis=0)
                gl_rg = np.append(gl_rg, gln.reshape(1, nx, nymax), axis=0)
                numerator_rg = np.append(numerator_rg, (numeratorn*np.exp(overflowfactorsn)).reshape(1, nx), axis=0)
                denominator_rg = np.append(denominator_rg, (denominatorn*np.exp(overflowfactorln)).reshape(1, nx), axis=0)
                deltag_rg = np.append(deltag_rg, deltagn.reshape(1, nx), axis=0)
                dgmix_rg = np.append(dgmix_rg, dgmixn.reshape(1, nx), axis=0)
                print(f'n = {n} : maximum error = {egn}, sum of error = {np.sum(np.abs(deltagn[1:-1] / dgmixn[1:-1]))}')
            if (egn < tol_dgmix): # ensure all ddgmix(x) < tolerance
                if iterationdata is True: 
                    print('Converged')
                break
            elif not np.isfinite(egn):
                print(f'Diveraged (egn={egn})')
                break
            elif (egn >= tol_dgmix) and (iteration is not None):
                if (n >= iteration - 1):
                    if iterationdata is True: 
                        print('Not converged (exceed the maximum iteration number)')
                    break

        if iterationdata is True: 
            self.gs_rg = gs_rg.copy()
            self.gl_rg = gl_rg.copy()
            self.numerator_rg = numerator_rg.copy()
            self.denominator_rg = denominator_rg.copy()
            self.__dgmix_rg = dgmix_rg.copy()
            self.deltag_rg = deltag_rg.copy()
            self.nid = n + 1
        else:
            dgmix_rg = dgmixn.reshape(1, nx)
            self.__dgmix_rg = dgmix_rg.copy()
            self.nid = 1
        self.iteration = n + 1
        
        return dgmix_rg

    def calc_dgmix_rgaw(self):
        """
        Parameters
        ----------
        xi : numpy array, float ncXnx
            mole fraction of component i  
        dxi : float
            differential mole fraction of component i  
        lhat0i : float
            one of Renormalization Group parameter 
        deltai : float
            one of Renormalization Group parameter 
        iteration : int or None
            total iteration time you want to calculate
        iterationdata : boolean
            Whether need to leave iteration data
        nx : int
            total number of mole fraction data
        nymax : int
            total number of mole fraction data for maximum varible y
        nid : int
            actual iteration data for gibbs free energy of mixing 
            (if iterationdata is True: iteration, else: 1)
        xmin : numpy array, float 1Xnx
            minimum mole fraction for all component in xi
            eg. if x[i] = np.array([0.3, 0.7]), xmin[i] = 0.3
        ymax : numpy array, float 1Xnymax
            minimum mole fraction for all component in xi
        ui : float
            one of Renormalization Group parameter (total mixing internal energy for x1=0.5)
        tol_dgmix : float
            the tolerance for Renormalization Group iteration 
        n : int
            iteration step
        d : int
            dimension
        dgmix0 : numpy array, float 1Xnx
            total molar Gibbs free energy of mixing divide by RT for all mole fraction data
        dgmixn : numpy array, float 1Xnx
            total molar Gibbs free energy of mixing divide by RT for all mole fraction data at nth iteration
        dgmixnm1 : numpy array, float 1Xnx
            total molar Gibbs free energy of mixing divide by RT for all mole fraction data at (n-1)th iteration
        dgmix_rg : None or numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            total molar Gibbs free energy of mixing divide by RT for all mole fraction data (calculated by liquid modol+RG)
        deltagn : numpy array, float 1Xnx
            correction to total molar Gibbs free energy of mixing divide by RT for all mole fraction data at nth iteration
        deltag_rg : numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            correction to total molar Gibbs free energy of mixing divide by RT for all mole fraction data 
            (calculated by liquid modol+RG)
            
        """
        
        if self.__dgmix_rgaw is not None: # dgmix_rg have been calculated
            return self.__dgmix_rgaw
        """Get class attributes"""
        xi = self.x
        dxi = self.dx
        lhat0i = self.lhat0
        deltai = self.delta
        deltali = self.deltal
        iteration = self.iteration
        iterationdata = self.iterationdata
        nx = self.nx
        jmx_tail = self.jmx_tail
        jpx_tail = self.jpx_tail
        jpx_end = self.jpx_end
        dgmix0 = self.dgmix_cg
        d = self.d
        ui = self.u  
        c_rg = self.c_rg
        tol_dgmix = self.tol_lle_dgmix[self.liquidmodelname]

        """Declare array"""

        if iterationdata is True:
            deltag_rg = np.zeros((1, nx), dtype=float)
            dgmix_rg = dgmix0.copy().reshape(1, nx)
        
        n = 0 
        dgmixn = dgmix0.copy()
        while True:
            n += 1
            dgmixnm1 = dgmixn.copy()

            # calculate deltagn by c++
            deltagn = calc_deltagn_cpp(nx, n, d, lhat0i, deltai, deltali, ui, c_rg, xi[0,:], jmx_tail, jpx_tail, jpx_end, dgmixnm1)
            
            dgmixn = dgmixnm1 + deltagn
            egn = np.max(np.abs(deltagn[1:-1]))
            # egn = np.max(np.abs(deltagn[1:-1] / dgmixn[1:-1]))
            if iterationdata is True: 
                deltag_rg = np.append(deltag_rg, deltagn.reshape(1, nx), axis=0)
                dgmix_rg = np.append(dgmix_rg, dgmixn.reshape(1, nx), axis=0)
                print(f'n = {n} : maximum error = {egn}, sum of error = {np.sum(np.abs(deltagn[1:-1] / dgmixn[1:-1]))}')
            if (egn < tol_dgmix): # ensure all ddgmix(x) < tolerance
                if iterationdata is True: 
                    print('Converged')
                break
            elif not np.isfinite(egn):
                print(f'Diveraged (egn={egn})')
                break
            elif (egn >= tol_dgmix) and (iteration is not None):
                if (n >= iteration - 1):
                    if iterationdata is True: 
                        print('Not converged (exceed the maximum iteration number)')
                    break

        if iterationdata is True: 
            self.__dgmix_rgaw = dgmix_rg.copy()
            self.deltag_rgaw = deltag_rg.copy()
            self.nid = n + 1
        else:
            dgmix_rg = dgmixn.reshape(1, nx)
            self.__dgmix_rgaw = dgmix_rg.copy()
            self.nid = 1
        self.iteration = n + 1
        
        return dgmix_rg
    
    @property
    def dgmix_rg(self):
        """
        Parameters
        ----------
        dgmix_rg : None or numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            total molar Gibbs free energy of mixing divide by RT for all mole fraction data (calculated by liquid modol+RG)
            
        """
        rgtype = self.rgtype
        if rgtype is None:
            return self.calc_dgmix_rg()
        elif rgtype == 'A':
            return self.calc_dgmix_rgaw()
    
    @property
    def ddgmixdx1_rg(self):
        """
        Parameters
        ----------
        ddgmixdx1_rg : numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            first derivatives of Gibbs free energy divide by RT (numerical equation)
        """
        xi = self.x
        dxi = self.dx
        dgmix_rg = self.dgmix_rg
        nid = self.nid
        ddgmixdx1_rg = np.zeros((nid, self.nx), dtype=float)
        for ix, xe in enumerate(xi.T):
            if ix == 0:
                ddgmixdx1_rg[:,ix] = ( dgmix_rg[:,ix+1] - dgmix_rg[:,ix] ) / dxi
            elif ( ix == self.nx - 1 ):
                ddgmixdx1_rg[:,ix] = ( dgmix_rg[:,ix] - dgmix_rg[:,ix-1] ) / dxi
            else:
                ddgmixdx1_rg[:,ix] = ( dgmix_rg[:,ix+1] - dgmix_rg[:,ix-1] ) / ( 2 * dxi )
        return ddgmixdx1_rg
    
    @property
    def d2dgmixdx12_rg(self):
        """
        Parameters
        ----------
        d2dgmixdx12_rg : numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            second derivatives of Gibbs free energy divide by RT (numerical equation)
        """
        xi = self.x
        dxi = self.dx
        dgmix_rg = self.dgmix_rg # order of dgmix_rg and nid cannot change
        nid = self.nid
        d2dgmixdx12_rg = np.zeros((nid, self.nx), dtype=float)
        for ix, xe in enumerate(xi.T):
            if ix == 0:
                d2dgmixdx12_rg[:,ix] = ( dgmix_rg[:,ix+2] - 2 * dgmix_rg[:,ix+1] + dgmix_rg[:,ix] ) / ( dxi **2 )
            elif ( ix == self.nx - 1 ):
                d2dgmixdx12_rg[:,ix] = ( dgmix_rg[:,ix-2] - 2 * dgmix_rg[:,ix-1] + dgmix_rg[:,ix] ) / ( dxi **2 )
            else:
                d2dgmixdx12_rg[:,ix] = ( dgmix_rg[:,ix+1] - 2 * dgmix_rg[:,ix] + dgmix_rg[:,ix-1] ) / ( dxi **2 )
        return d2dgmixdx12_rg
    
    @property
    def d3dgmixdx13_rg(self):
        """
        Parameters
        ----------
        d3dgmixdx13_rg : numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            third derivatives of Gibbs free energy divide by RT (numerical equation)
        """
        xi = self.x
        dxi = self.dx
        dgmix_rg = self.dgmix_rg # order of dgmix_rg and nid cannot change
        nid = self.nid
        d3dgmixdx13_rg = np.zeros((nid, self.nx), dtype=float)
        for ix, xe in enumerate(xi.T):
            if ix == 0:
                d3dgmixdx13_rg[:,ix] = ( dgmix_rg[:,ix+4] - 2 * dgmix_rg[:,ix+3] + 2 * dgmix_rg[:,ix+1] - dgmix_rg[:,ix] ) / ( 2 * dxi **3 )
            elif ix == 1:
                d3dgmixdx13_rg[:,ix] = ( dgmix_rg[:,ix+3] - 2 * dgmix_rg[:,ix+2] + 2 * dgmix_rg[:,ix] - dgmix_rg[:,ix-1] ) / ( 2 * dxi **3 )
            elif ( ix == self.nx - 2 ):
                d3dgmixdx13_rg[:,ix] = ( dgmix_rg[:,ix+1] - 2 * dgmix_rg[:,ix] + 2 * dgmix_rg[:,ix-2] - dgmix_rg[:,ix-3] ) / ( 2 * dxi **3 )
            elif ( ix == self.nx - 1 ):
                d3dgmixdx13_rg[:,ix] = ( dgmix_rg[:,ix] - 2 * dgmix_rg[:,ix-1] + 2 * dgmix_rg[:,ix-3] - dgmix_rg[:,ix-4] ) / ( 2 * dxi **3 )
            else:
                d3dgmixdx13_rg[:,ix] = ( dgmix_rg[:,ix+2] - 2 * dgmix_rg[:,ix+1] + 2 * dgmix_rg[:,ix-1] - dgmix_rg[:,ix-2] ) / ( 2 * dxi **3 )
        return d3dgmixdx13_rg
    
    @property
    def g_cg(self):
        """
        Parameters
        ----------
        g_cg : None or numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            total molar Gibbs free energy divide by RT for all mole fraction data (calculated by liquid modol)
            
        """
        ti = self.t
        xi = self.x
        epsilon11dk = self.epsilon11dk
        epsilon22dk = self.epsilon22dk
        Nc = self.Nc
        dgmix_cg = self.dgmix_cg
        g_cg = dgmix_cg + ( xi[0] * epsilon11dk + xi[1] * epsilon22dk ) * Nc / ( 2 * ti ) 
        return g_cg
    
    @property
    def g_rg(self):
        """
        Parameters
        ----------
        g_rg : None or numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            total molar Gibbs free energy divide by RT for all mole fraction data (calculated by liquid modol+RG)
            
        """
        ti = self.t
        xi = self.x
        epsilon11dk = self.epsilon11dk
        epsilon22dk = self.epsilon22dk
        Nc = self.Nc
        dgmix_rg = self.dgmix_rg
        g_rg = dgmix_rg + ( xi[0] * epsilon11dk + xi[1] * epsilon22dk ) * Nc / ( 2 * ti ) 
        return g_rg
    
    def calc_gt_cg(self): # G/Nk
        ti = self.t
        g_cg = self.g_cg
        gt_cg = g_cg * ti
        return gt_cg
    
    def calc_gt_rg(self): # G/Nk
        ti = self.t
        g_rg = self.g_rg
        gt_rg = g_rg * ti
        return gt_rg
    
    @property
    def mu_rg(self):
        """
        Parameters
        ----------
        mu_rg : numpy array, float (if iterationdata is True: nXncXnx, else: 1XncXnx) 
            chemical potential for component i in the mixture divide by RT (numerical equation)
        """
        xi = self.x
        nxi = self.nx
        nci = self.nc
        dgmix_rg = self.dgmix_rg
        nid = self.nid
        ddgmixdx1_rg = self.ddgmixdx1_rg
        mu_rg = np.zeros((nid, nci, nxi), dtype=float)
        for n in range(nid):
            mu_rg[n,0,:] = dgmix_rg[n,:] + xi[1] * ddgmixdx1_rg[n,:]
            mu_rg[n,1,:] = dgmix_rg[n,:] - xi[0] * ddgmixdx1_rg[n,:]
        return mu_rg
        
    @property
    def gammam_rg(self):
        """
        Parameters
        ----------
        gammam_rg : numpy array, float (if iterationdata is True: nXncXnx, else: 1XncXnx) 
            activity coefficient for component i in the mixture (numerical equation)
        """
        xi = self.x
        nxi = self.nx
        nci = self.nc
        mu_rg = self.mu_rg
        nid = self.nid
        gammam_rg = np.zeros((nid, nci, nxi), dtype=float)
        for ic in range(nci):
            gammam_rg[:,ic,:] = np.exp(mu_rg[:,ic,:]) / xi[ic]
        return gammam_rg
    
    @property
    def cp_cg(self):
        """
        Parameters
        ----------
        cp_cg : None or numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            total molar heat capacity divide by R for all mole fraction data (calculated by liquid modol)
            
        """
        ti = self.t
        dt = self.dt
        d2gtdt2_cg = self.calc_d2gtdt2_cg(dt)
        cp_cg = - ti * d2gtdt2_cg
        return cp_cg
    
    @property
    def cp_rg(self):
        """
        Parameters
        ----------
        cp_rg : None or numpy array, float (if iterationdata is True: nXnx, else: 1Xnx) 
            total molar heat capacity divide by R for all mole fraction data (calculated by liquid modol+RG)
            
        """
        ti = self.t
        dt = self.dt
        d2gtdt2_rg = self.calc_d2gtdt2_rg(dt)
        cp_rg = - ti * d2gtdt2_rg
        return cp_rg
    
    def calc_xec0(self):
        # get extreme value composition(first derivative of dgmix for x1 is zero) at zero iteration 
        xi = self.x
        ddgmix0dx1 = self.ddgmixdx1_cg
        nxi = self.nx
        nxti = nxi - 1
        nxhalf = nxti // 2
        nci = self.nc
        xeca0 = np.zeros(nci, dtype=float)
        xecb0 = np.zeros(nci, dtype=float)
        flag_xeca = False
        flag_xecb = False
        for ix in range(nxi):
            if ( ddgmix0dx1[ix] >= 0 ) and ( flag_xecb is False ):
                xecb0 = xi[:,ix]
                flag_xecb = True
            if ( ddgmix0dx1[nxti-ix] <= 0 ) and ( flag_xeca is False ):
                xeca0 = xi[:,nxti-ix]
                flag_xeca = True
            if ( flag_xeca is True ) and ( flag_xecb is True ):
                break
            elif ( ix > nxhalf ) and ( flag_xeca is False ) and ( flag_xecb is False ):
                xeca0 = np.full(nci, np.nan)
                xecb0 = np.full(nci, np.nan)
                break
        else:
            xeca0 = np.full(nci, np.nan)
            xecb0 = np.full(nci, np.nan)

        if xeca0[0] - xecb0[0] < 0.: # T>Tc
            xeca0 = np.full(nci, np.nan)
            xecb0 = np.full(nci, np.nan)
        self.xeca0 = xeca0
        self.xecb0 = xecb0 

    '''
    def calc_xec0(self):
        # get extreme value composition(first derivative of dgmix for x1 is zero) at zero iteration 
        xi = self.x
        dxi = self.dx
        iteration = self.iteration
        nci = self.nc
        ddgmix0dx1 = self.ddgmixdx1_cg
        xeca0 = np.zeros(nci, dtype=float)
        xecb0 = np.zeros(nci, dtype=float)
        for (xe, ddgmix0dx1e) in zip(xi.T, ddgmix0dx1):
            if ddgmix0dx1e >= 0:
                xecb0 = xe
                break
            else:
                pass
        for (xe, ddgmix0dx1e) in zip(np.flipud(xi.T), np.flipud(ddgmix0dx1)):
            if ddgmix0dx1e <= 0:
                xeca0 = xe
                break
            else:
                pass
        if xeca0[0] - xecb0[0] < 0.: # T>Tc
            xeca0 = np.full(nci, np.nan)
            xecb0 = np.full(nci, np.nan)
        return xeca0, xecb0    
    '''
    def calc_xic(self):
        # get inflection composition (second derivative of dgmix for x1 is zero)
        xi = self.x
        d2dgmixdx12_rg = self.d2dgmixdx12_rg
        nxi = self.nx
        nxti = nxi - 1
        nxhalf = nxti // 2
        nci = self.nc
        nid = self.nid
        #nica = np.zeros(nid, dtype=int)
        #nicb = np.zeros(nid, dtype=int)
        xica = np.zeros((nci, nid), dtype=float)
        xicb = np.zeros((nci, nid), dtype=float)
        tol = 0.
        for n in range(nid):
            flag_xica = False
            flag_xicb = False
            for ix in range(nxi): 
                if ( d2dgmixdx12_rg[n,ix] <= tol ) and ( flag_xicb is False ):
                    xicb[:,n] = xi[:,ix]
                    #nicb[n] = ix
                    flag_xicb = True
                if ( d2dgmixdx12_rg[n,nxti-ix] <= tol ) and ( flag_xica is False ):
                    xica[:,n] = xi[:,nxti-ix]
                    #nica[n] = nxti-ix
                    flag_xica = True
                if ( flag_xica is True ) and ( flag_xicb is True ):
                    break
                elif ( ix > nxhalf ) and ( flag_xica is False ) and ( flag_xicb is False ):
                    xica[:,n] = np.full(nci, np.nan)
                    xicb[:,n] = np.full(nci, np.nan)
                    break
            else:
                xica[:,n] = np.full(nci, np.nan)
                xicb[:,n] = np.full(nci, np.nan)
        self.xica = xica
        self.xicb = xicb
        #self.nica = nica
        #self.nicb = nicb
        self.xa = xica[:,-1]
        self.xb = xicb[:,-1]
    
    def calc_critical_tx_objfun(self, x):
        ti = x
        self.set_t(ti)
        self.calc_xic()
        x1ai = self.xa[0]
        x1bi = self.xb[0]
        ex = x1ai - x1bi
        return ex
    
    def calc_critical_tx(self, ucp=True, lcp=False, tgui=None, tgli=None, t_stepgu=None, t_stepgl=None):
        nci = self.nc
        dxi = self.dx
        tol_lle_dgmix = self.tol_lle_dgmix[self.liquidmodelname]
        tol_tc = 1e-14
           
        def solve_cp(tsi, t_step):
            toi = tsi
            t_stepo = t_step
            t_stepn = t_step
            exo = self.calc_critical_tx_objfun(toi)
            while np.isnan(exo): # if exo == nan (single phase), change initial temperature until exo > 0 (phase separation)
                toi = toi * 0.9 if t_step > 0 else toi * 1.1
                exo = self.calc_critical_tx_objfun(toi)
            while True:
                tni = toi + t_stepn
                exn = self.calc_critical_tx_objfun(tni)
                # print(tni, exn, t_stepn)
                if ( exn >= dxi ): # T<Tc
                    toi = tni  
                    if np.isnan(exo): # the ex of previous step is nan
                        t_stepn = 0.5 * t_stepo
                    else: # the ex of previous step is not nan
                        t_stepn = t_stepo
                    if ( abs(t_stepn) <= tol_tc ):
                        print('Not found CP')
                        tci = tni
                        xci = np.full(nci, np.nan)
                        break
                elif np.isnan(exn): # T>Tc
                    t_stepn = 0.5 * t_stepo
                else: # T=Tc
                    tci = tni
                    xci = self.xa
                    break
                exo = exn
                t_stepo = t_stepn
            # print('converged', tci, xci)
            return tci, xci
        '''
        if ( ucp is True ) or ( lcp is True ):
            
            if ( tgui is None ) or ( tgli is None ):
                t_test = np.array([200., 250., 300., 350., 400., 450.])
                ex_test = np.array([])
                for tti in t_test:
                    ex_test = np.append(ex_test, self.calc_critical_tx_objfun(tti))
        '''
        if ucp is True:
            tsu = ( tgui - 50 ) if tgui is not None else 280 # K
            t_stepu = t_stepgu if t_stepgu is not None else 100
            tcu, xcu = solve_cp(tsu, t_stepu)
        else:
            tcu, xcu = np.full(2, np.nan)
        if lcp is True:
            tgl = ( tgli + 20 ) if ( tgli is not None ) else ( tcu - 90 ) if ( ucp is True ) else 380 # K
            t_stepl = t_stepgl if t_stepgl is not None else - 50
            tcl, xcl = solve_cp(tgl, t_stepl)
        else:
            tcl, xcl = np.full(2, np.nan)
        tc = {'UCP':tcu, 'LCP':tcl}
        xc = {'UCP':xcu, 'LCP':xcl}
        return tc, xc

    def calc_critical_t_ising_objfun(self, x):
        ti = x[0]
        self.set_t(ti)
        cp_rgi = self.cp_rg[-1,500]
        errori = np.exp( - cp_rgi / 2 )
        # print(ti[0], cp_rgi, errori)
        return errori
    
    def calc_critical_t_ising(self, tgui=None, bounds=None):
        bounds = (2, 4) if bounds is None else bounds
        tgui = ( bounds[0] + bounds[1] ) / 2 if tgui is None else tgui
        toi = self.t
        iterationoi = self.iteration
        self.set_rgp(iteration=self.nmax)
        resi = least_squares(fun=self.calc_critical_t_ising_objfun, x0=tgui, bounds=bounds, diff_step=1.e-4)
        tc = resi.x[0]
        self.set_t(toi)
        self.set_rgp(iteration=iterationoi)
        return tc
        
        
"""The functions to solve BIP for NRTL model at critical point"""
def calc_critical_nrtl_bip_objfun(x, *data):
    lmobject, ti, xi, aij, cij, dij, eij, fij = data
    b12, b21 = x
    bij = np.array([[0, b12],[b21, 0]])
    lmobject.set_tx(ti, xi)
    lmobject.set_bip(aij, bij, cij, dij, eij, fij)
    # The equations below are solved simultaneously to find the critical temperature and composition
    f1 = lmobject.calc_dmudxa()[0,0]
    f2 = lmobject.calc_d2mudx2a()[0,0]
    return f1, f2
    
def calc_critical_nrtl_bip(tci, xci):
    liquidmodelnamei = 'NRTL'
    cij = np.array([[0., 0.2],
                    [0.2, 0.]])
    aij = bij = dij = eij = fij = np.zeros(np.shape(cij), dtype=float)
    g = [300, 300] # Initial guess
    ccnb_lmobject = lm.Context(lm.create(liquidmodelnamei))
    data = (ccnb_lmobject, tci, xci, aij, cij, dij, eij, fij)
    b12, b21 = fsolve(calc_critical_nrtl_bip_objfun, g, args=data, xtol=Calc_binary_critical.tol_critical[liquidmodelnamei])
    return b12, b21
